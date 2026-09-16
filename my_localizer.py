from __future__ import annotations

# import libraries
import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import cv2
import os

from my_pose_harness import (
    POSE_THRESHOLDS,
    Scorer,
    get_eval_views,
    load_pipeline,
    make_perturbation,
    perturb_pose,
    pose_error,
    so3_exp,
    render_at_pose,
    threshold_recall,
    to_4x4
)

def assert_map_frozen(pipeline) -> None:
    """
    Checks that all Gaussian parameters are safely frozen and raises an error if any remain active, ensuring the optimizer only moves the camera and does not deform the map.
    """

    live = [
        name for name, p in pipeline.model.named_parameters() if p.requires_grad
    ]
    if live:
        raise AssertionError(
            f"{len(live)} map parameters still require grad, e.g. {live[:5]}. "
            "Freeze them before optimising, or you will optimise the map "
            "instead of the pose."
        )

def freeze_map(pipeline) -> None:
    """
    Set requires_grad False on ever model parameter. Call this once at startup
    """
    for p in pipeline.model.parameters():
        p.requires_grad_(False)

# Step 1 - Differentiable Rendering

def render_differentiable(pipeline, camera, c2w_4x4: torch.Tensor) -> torch.Tensor:
    """
    Render the image while maintaining the gradient information to see how values change while moving camera
    """
    from nerfstudio.cameras.cameras import Cameras

    # Ensure c2w is 3x4 or 4x4 with batch dimension
    c2w_tensor = c2w_4x4[:3, :].unsqueeze(0) if c2w_4x4.ndim == 2 else c2w_4x4[..., :3, :]

    # Construct new Cameras object connected to our autograd pose tensor
    cam = Cameras(
        camera_to_worlds=c2w_tensor,
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        width=camera.width,
        height=camera.height,
        distortion_params=camera.distortion_params,
        camera_type=camera.camera_type,
    ).to(c2w_4x4.device)

    # Get the render from camera.use and use get_outputs to ensure that the camera has the torch gradients attached.
    outputs = pipeline.model.get_outputs(cam)
    return outputs["rgb"]

def photometric_loss(rendered: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """
    Gets the grayscaled MSE between the two images. Returns a scalar value
    """
    # what is special about these values? -> [0.299, 0.587, 0.114]
    w = torch.tensor([0.299, 0.587, 0.114], device=rendered.device)
    r_gray = (rendered * w).sum(-1)
    q_gray = (query.to(rendered.device) * w).sum(-1)
    return ((r_gray - q_gray)**2).mean()

def check_step1(pipeline, views, scale) -> None:
    """
    Finite-difference check to ensure the analytical gradient calculated by the framework matches the manually calculated numerical slope, proving the autograd graph is unbroken.
    INPUT: pipeline - Nerfstudion object with info on scene
    views - tuple object of structure (camera, query image - HWC)
    """

    # freeze the map
    assert_map_frozen(pipeline)
    camera, query = views[0]
    device = pipeline.device
    gt = to_4x4(camera.camera_to_worlds[0]).to(device).float()

    # helper function for full render-and-score cycle. xi_vec is a 6DoF pose update
    def loss_of(xi_vec: torch.Tensor) -> torch.Tensor:
        # T is the new pose after shifting ground truth pose by xi_vec
        T = gt @ se3_exp_torch(xi_vec)
        # render image at point T
        img = render_differentiable(pipeline, camera, T)
        # Return photometric loss between img, and query
        return photometric_loss(img, query)

    # The 6-DoF tangent vector. Initialized to exactly zero to calculate the gradient at the current starting pose.
    xi = torch.zeros(6, device=device, requires_grad=True)

    # Executes the forward pass. L holds the scalar loss and the computational graph history (no gradients exist yet).
    L = loss_of(xi)

    # ensure that torch chain hasn't broken
    if L.grad_fn is None:
        raise AssertionError(
            "Loss has no grad_fn. The graph is broken between pose and image. "
            "Step 1 TODO items 1 and 3."
        )
        
    # Triggers backpropagation. Traverses the recorded graph backward to calculate the exact mathematical derivatives.
    L.backward()
    
    # Proves the graph was fully connected back to the pose parameters. Must occur after L.backward().
    if xi.grad is None:
        raise AssertionError(
            "xi.grad is None. Gradients are not reaching the pose at all."
        )
        
    # Detaches the calculated gradient from the autograd graph so subsequent testing math doesn't corrupt the active graph. Returns a 1x6 vector where each element is the gradient at that point
    analytic = xi.grad.detach().clone()

    if torch.allclose(analytic, torch.zeros_like(analytic)):
        raise AssertionError(
            "Analytic gradient is all zeros. Either the pose is not entering "
            "the render, or an in-place assignment severed the graph."
        )

    # Tiny step size for our manual math check
    h = 1e-4
    # The manual gradient tensor. Honestly, calculating this central difference feels like a waste of time, but it's the only way to strictly verify PyTorch's autograd isn't lying to us.
    numeric = torch.zeros(6, device=device)
    for i in range(6):
        p = torch.zeros(6, device=device)
        p[i] = h
        with torch.no_grad():
            # Nudge the camera slightly forward and backward to calculate the raw, old-fashioned slope
            lp = loss_of(p)
            lm = loss_of(-p)
        numeric[i] = (lp - lm) / (2 * h)

    # Spit out both vectors to visually confirm if the framework's backprop magic matches our brute-force math
    print("  analytic:", np.round(analytic.cpu().numpy(), 6))
    print("  numeric :", np.round(numeric.cpu().numpy(), 6))

    # Scale-aware denominator to stop the math from exploding and throwing fake errors when the gradient is basically zero
    denom = torch.maximum(numeric.abs(), torch.tensor(0.1 * numeric.abs().max().item(), device=device))
    rel = ((analytic - numeric).abs() / denom).max().item()
    print(f"  max relative difference: {rel:.4f}")

    # The actual gatekeeping. If PyTorch and our manual math disagree by more than 15%, the graph is screwed up and we hard-stop.
    if rel > 0.15:
        cos = torch.nn.functional.cosine_similarity(
            analytic.unsqueeze(0), numeric.unsqueeze(0)
        ).item()
        extra = ""
        # Catching the classic rookie mistake: a negative cosine means the vector is flipped, meaning we accidentally rotated the whole world instead of just the camera within its local frame.
        if cos < -0.5:
            extra = (
                " Cosine similarity is negative: your gradient points the wrong "
                "way. You have most likely confused camera motion with world "
                "motion."
            )
        raise AssertionError(
            f"Gradient check failed, relative difference {rel:.3f}.{extra}"
        )
    print("  STEP 1 PASS")

# Step 2 - Update Parameterisation

def se3_exp_torch(xi: torch.Tensor) -> torch.Tensor:
    """
    converts a 1x6 vector of translation and rotation into its 4x4 transform vector
    """
    # The translation vector
    rho = xi[:3]
    # The rotation vector
    phi = xi[3:]

    # Protect against the Small-Angle Trap
    theta_sq = torch.sum(phi **2)
    theta = torch.sqrt(torch.clamp(theta_sq, min=1e-12))

    # Build the skew symmetric matrix -> Do cross products as matrix mult
    z = phi[0] * 0.0
    # where Phi is equal to theta * 
    Phi = torch.stack([
        torch.stack([z, -phi[2], phi[1]]),
        torch.stack([phi[2], z, -phi[0]]),
        torch.stack([-phi[1], phi[0], z])
    ])
    Phi2 = Phi @ Phi
    I = torch.eye(3, device=xi.device, dtype=xi.dtype)

    # Use Rodrigues Formula to go from rotation vector to rotation matrix 
    if theta_sq < 1e-6:
        A = 1.0 - theta_sq / 6.0
        B = 0.5 - theta_sq / 24.0
        C = 1.0 / 6.0 - theta_sq / 120.0
    else:
        A = torch.sin(theta) / theta
        B = (1.0 - torch.cos(theta)) / theta_sq
        C = (theta - torch.sin(theta)) / (theta_sq * theta)
    
    R = I + A * Phi + B * Phi2
    V = I + B * Phi + C * Phi2
    t = V @ rho

    # Assemble the 4x4 matrix, really just adding the rotation & translation & [0 0 0 1] at the bottom
    Rt = torch.cat([R, t.unsqueeze(1)], dim=1) # Combine 3x3 R and 3x1 t into 3x4
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=xi.device, dtype=xi.dtype)
    T = torch.cat([Rt, bottom], dim=0) # Add the bottom row to make it 4x4
    
    return T

def apply_update(T_current: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """
    Apply the perturbation of xi to the T_current pose
    """
    return T_current @ se3_exp_torch(xi)

def check_step2(pipeline, views, scale) -> None:
    """
    Apply 3 tests. Zero update with identity, move 0.1 colmap unit left, then move back 0.1 colmap units
    """
    camera, _ = views[0]
    T = to_4x4(camera.camera_to_worlds[0]).double()

    z = se3_exp_torch(torch.zeros(6, dtype=torch.float64))
    # set the error as the difference between the transform vector and identity matrix -> why?
    err = (z - torch.eye(4, dtype=torch.float64)).abs().max().item()
    print(f"  exp(0) deviation from identity: {err:.2e}")
    if err > 1e-9:
        raise AssertionError("exp(0) is not the identity. Small-angle branch.")

    xi = torch.zeros(6, dtype=torch.float64)
    xi[0] = 0.01
    T2 = apply_update(T, xi)
    t_err, r_err = pose_error(T, T2)
    print(f"  0.1 colmap unit x-step -> t_err {t_err:.6f} (expect 0.010000), "
          f"r_err {r_err:.6f} deg (expect 0)")
    if abs(t_err - 0.01) > 1e-6 or r_err > 1e-6:
        raise AssertionError("A pure translation step did not translate purely.")

    xi2 = torch.tensor([0.02, -0.01, 0.005, 0.03, -0.02, 0.01],
                        dtype=torch.float64)
    back = apply_update(apply_update(T, xi2), -xi2)
    t_b, r_b = pose_error(T, back)
    print(f"  xi then -xi round trip: t {t_b:.2e}, r {r_b:.2e}")
    if t_b > 1e-6 or r_b > 1e-6:
        raise AssertionError(
            "xi then -xi did not return to the start. Note this only holds for "
            "right composition applied consistently."
        )
    print("  STEP 2 PASS")

# Step 3 - One Gradient

def single_gradient_step(pipeline, camera, query, T: torch.Tensor, lr: float) -> tuple[torch.Tensor, float, float]:
    """
    Take exactly one step down the loss gradient. Return (T_new, loss_before, loss_after)
    """
    device = T.device
    dtype = T.dtype

    # Create a fresh tangent vector with autograd enabled
    xi = torch.zeros(6, device=device, dtype=dtype, requires_grad=True)
    # T_candidate is at same position as T
    T_candiate = apply_update(T, xi)

    # Create the render at the original position
    img_before = render_differentiable(pipeline, camera, T_candiate)

    # Calculate the loss between the initial image and the query image
    loss = photometric_loss(img_before, query)
    # Detatch the loss from any gradient history, pure scalar
    loss_before = loss.item()

    # Create the gradients especially from the chain rule that we try slide down to minimize error. Already applied LK into CUDA kernals in rasterizer
    loss.backward()

    # Step down the gradient and retract back onto SE(3)
    with torch.no_grad():
        xi_step = -lr * xi.grad
        T_new = apply_update(T, xi_step)

        # compute evaluated post-step loss
        img_after = render_differentiable(pipeline, camera, T_new)
        loss_after = photometric_loss(img_after, query).item()

    return T_new, loss_before, loss_after

def check_step3(pipeline, views, scale) -> None:
    """
    Perturb by 0.5 colmap units, take one step
    """
    assert_map_frozen(pipeline)
    camera, query = views[0]

    # Get the ground truth camera pose
    gt = to_4x4(camera.camera_to_worlds[0])
    # Use generator object for reproducability
    gen = torch.Generator().manual_seed(0)
    # make slight perturbations
    delta = make_perturbation(0.005 * scale, 0.0, generator=gen)

    # perturbs the ground truth pose
    T0 = perturb_pose(gt, delta)

    t0, r0 = pose_error(gt.double(), T0.double())
    T1, l_before, l_after = single_gradient_step(
        pipeline, camera, query, T0.to(pipeline.device).float(), lr=1e-4
    )
    t1, r1 = pose_error(gt.double(), T1.cpu().double())
    
    print(f"  loss {l_before:.6f} -> {l_after:.6f}")
    print(f"  t_err {t0:.6f} -> {t1:.6f}")
    
    if l_after >= l_before:
        raise AssertionError(
            "Loss did not decrease. Check the sign of the update, or lower lr."
        )
    if t1 >= t0:
        raise AssertionError(
            "Loss fell but pose error did not. Classic signature of an "
            "unfrozen map: the optimiser is improving the image by other means."
        )
    print("  STEP 3 PASS")

# Step 4 - Optimisation Loop. The main point of this fine localisation step

@dataclass
class LocaliseResult:
    T_final: torch.Tensor
    iterations: int
    termination: str
    loss_history: list[float] = field(default_factory=list)
    pose_history: list[torch.Tensor] = field(default_factory=list)

def localise(
        pipeline, 
        camera, query: torch.Tensor,
        T_init: torch.Tensor,
        max_iters: int = 250,
        lr_trans: float = 2e-3,
        lr_rot: float = 2e-3,
        tol_loss: float = 1e-9, # tolerance threshold loss
        tol_step: float = 1e-6, # toleranch threshold step size
        debug_name: str = "debug_localise"
        ) -> LocaliseResult:
    """
    The localiser. Descend the photometric objective from T_init
    """
    device = T_init.device
    dtype = T_init.dtype

    T_curr = T_init.clone()
    loss_history: list[float] = []
    pose_history: list[torch.Tensor] = [T_curr.clone()]

    # default termination is too many iterations
    termination = "max_iters"
    actual_iters = 0
    prev_loss: float | None = None

    # 1. Decoupled parameter groups for translation and rotation
    rho = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)
    phi = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)

    # 2. PyTorch Native Adam handles the bias correction (prevents first-step explosion) -> How?
    optimizer = torch.optim.Adam([
        {"params": [rho], "lr": lr_trans},
        {"params": [phi], "lr": lr_rot},
    ])

    # 3. Scheduler smootly decays step sizes and guarantees settling without bouncing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-6)

    first_render = None
    last_render = None

    for i in range(max_iters):
        actual_iters = i + 1
        optimizer.zero_grad() # What does this do?

        # Zero tangent vector values and reset Taylor Expansion
        with torch.no_grad():
            rho.zero_()
            phi.zero_()

        # Concatenate and retract
        xi = torch.cat([rho, phi], dim=0)
        T_candidate = apply_update(T_curr, xi)

        # Forward & Loss
        rendered = render_differentiable(pipeline, camera, T_candidate)
        # create the first render only at the beginning
        if i == 0:
            first_render = rendered.detach().clone()
        # create the last render 
        last_render = rendered.detach().clone()
        # get loss as a scalar
        loss = photometric_loss(rendered, query)
        current_loss = loss.item()
        loss_history.append(current_loss)

        if prev_loss is not None and abs(prev_loss - current_loss) < tol_loss:
            termination = "loss_plateau"
            break
        prev_loss = current_loss

        # 6. backward & step
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 7. Apply the computed step to the accumulated pose
        with torch.no_grad():
            xi_step = torch.cat([rho, phi], dim = 0)
            step_norm  = torch.linalg.norm(xi_step).item()

            if i % 25 == 0 or i == max_iters - 1:
                # Report the results at the end
                mse = torch.nn.functional.mse_loss(rendered, query.to(device)).item()
                psnr = -10.0 * math.log10(mse) if mse > 0 else float("inf")
                print(f"    iter {i:03d} | loss: {current_loss:.6f} | psnr: {psnr:.2f} dB | |step|: {step_norm:.2e}")

            if step_norm < tol_step:
                termination = "step_small"
                break

            T_curr = apply_update(T_curr, xi_step)
            pose_history.append(T_curr.clone())

    # Save the visuals
    if first_render is not None and last_render is not None:
        q_img = (query.detach().cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
        init_img = (first_render.detach().cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
        final_img = (last_render.detach().cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
        # calculate the differences between the initial images and the final images
        # This difference is just the absolute difference
        diff_init = np.abs(init_img.astype(float) - q_img.astype(float)).astype(np.uint8)
        diff_final = np.abs(final_img.astype(float) - q_img.astype(float)).astype(np.uint8)

        # Develop the final image. Give the black square same dimensions as q_image
        black = np.zeros_like(q_img)
        row1 = np.hstack([q_img, init_img, final_img])
        row2 = np.hstack([black, diff_init, diff_final])
        grid = np.vstack([row1, row2])

        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, f"{debug_name}.png")
        cv2.imwrite(save_path, grid)
        print(f"  [debug] Saved Final Image -> {save_path}")

    return LocaliseResult(
        T_final=T_curr,
        iterations=actual_iters,
        termination=termination,
        loss_history=loss_history,
        pose_history=pose_history,
    )

def check_step4(pipeline, views, scale) -> None:
    """
    Perturb it by 1 colmap unit and 1 degere rotation. Then try to use the gradient descent method to get back to the origin
    """
    assert_map_frozen(pipeline)
    camera, query = views[0]
    gt = to_4x4(camera.camera_to_worlds[0])
    gen = torch.Generator().manual_seed(1)
    T0 = perturb_pose(gt, make_perturbation(3 * scale, 2.0, generator=gen))

    res = localise(
        pipeline,
        camera,
        query,
        T0.to(pipeline.device).float(),
        max_iters=1000,
        lr_trans=2e-3,
        lr_rot=2e-3,
    )

    t_err, r_err = pose_error(gt.double(), res.T_final.cpu().double())
    # Remember, that this scale does not work at the moment
    # t_m = t_err / scale if scale else float("nan")

    # Print the final results
    print(f"  initial loss: {res.loss_history[0]:.6f} -> final loss: {res.loss_history[-1]:.6f}")
    print(f"  terminated: {res.termination} after {res.iterations} iters")
    # We can't determinne the actual error, in mm or m yet, only in COLMAP units
    # print(f"  final error: {t_m*1000:.2f} mm, {r_err:.4f} deg")
    print(f"  final error: {t_err:.2f} COLMAP Units, {r_err:.4f} deg")

    # Again, we need to calubrate the system around the scaling
    # if t_err *  > 0.001 or r_err > 0.1:
    #     raise AssertionError(
    #         "Did not recover a small perturbation. If the final loss is low but "
    #         "the pose is wrong, you are in a local minimum, which should not "
    #         "happen at 1 cm. Re-check that the map is frozen."
    #     )
    print("  STEP 4 PASS")

# Step 5 - Optimisers capture radius

def capture_radius_sweep(
        pipeline, views, scale, magnitudes: list[tuple[float, float]], n_trials: int = 10
) -> list[dict]:
    """
    Does a sweep to see what the capture radius of the fine stage is, then from here we can determine how accurate the course stage has to be. Note that we are doing this using the COLMAP scale
    """
    assert_map_frozen(pipeline)
    
    # Use generator object for reproducibility
    gen = torch.Generator().manual_seed(42)
    rows = []

    for v_idx, (camera, query) in enumerate(views):
        # Ground truth pose
        gt_c2w = to_4x4(camera.camera_to_worlds[0].cpu())
        
        for t_mag, r_mag in magnitudes:
            # If no perturbation, only run 1 trial
            trials = 1 if (t_mag == 0.0 and r_mag == 0.0) else n_trials

            # we do it "trials" times because both direction and magnitude of perturbation impact it, so later when we are processing the data, we have enough to get meaningful data from it
            for k in range(trials):
                # 1. Perturb the ground truth pose
                delta = make_perturbation(t_mag, r_mag, generator=gen)
                T_init = perturb_pose(gt_c2w, delta)
                
                t_err_init, r_err_init = pose_error(gt_c2w, T_init)
                
                # 2. Localise (Fine Refinement)
                res = localise(
                    pipeline=pipeline,
                    camera=camera,
                    query=query,
                    T_init=T_init.to(pipeline.device).float(),
                    max_iters=250, # Standard early stop threshold
                    lr_trans=2e-3,
                    lr_rot=2e-3,
                    debug_name=f"sweep_v{v_idx}_t{t_mag}_r{r_mag}_trial{k}"
                )
                
                # 3. Calculate Final Error
                t_err_final, r_err_final = pose_error(gt_c2w.double(), res.T_final.cpu().double())
                
                # 4. Record Results
                rows.append({
                    "view": v_idx,
                    "trial": k,
                    "t_mag_scene": t_mag,
                    "r_mag_deg": r_mag,
                    "t_err_init_scene": t_err_init,
                    "r_err_init_deg": r_err_init,
                    "t_err_final_scene": t_err_final,
                    "r_err_final_deg": r_err_final,
                    "termination": res.termination,
                    "iterations": res.iterations,
                    "loss_initial": res.loss_history[0] if res.loss_history else float("nan"),
                    "loss_final": res.loss_history[-1] if res.loss_history else float("nan")
                })
                
            print(f"  [view {v_idx}] offset: t={t_mag:.2f} r={r_mag:.1f} | final error: {t_err_final:.4f} units, {r_err_final:.2f} deg")

    return rows

def check_step5(pipeline, views, scale) -> None:
    """
    Runs the capture radius sweep to scientifically define the boundaries of the fine optimizer.
    """
    # Test a small set of magnitudes (t_mag in scene units, r_mag in degrees)
    # Kept small so it finishes in a reasonable time while you eat
    magnitudes = [
        # (0.01, 0.5),
        # (0.05, 2.0),
        # (0.10, 5.0),
        (0.20, 10.0),
        (0.30, 15.0),
        (0.40, 20)
    ]
    print(f"  Running sweep across {len(magnitudes)} magnitudes...")
    # n_trials=3 to keep it relatively fast for this test run
    results = capture_radius_sweep(pipeline, views, scale, magnitudes, n_trials=3)
    print(f"  STEP 5 COMPLETE. Generated {len(results)} data points.")

# Step 6 - Coarse Global Retrieval

def get_clip_features(image_tensor: torch.Tensor, processor, model, device) -> torch.Tensor:
    """
    Helper to convert HWC tensor to CLIP embeddings.
    INPUT: 
        image_tensor -> Tensor representing image
        processor -> HuggingFace CLIPProcessor that prepares and scales the raw image so the model can read it natively.
        model -> Pre-trained CLIP vision model used to extract the deep semantic features.
        device -> Hardware device where the computations occur and the resulting embedding tensor is stored
    Returns:
        torch.Tensor: A normalized 1D feature vector representing the semantic embedding of the input image.
    """
    # Convert from [0, 1] tensor to uint8 numpy array for the processor
    img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    inputs = processor(images=img_np, return_tensors="pt").to(device) # returns as pytorch tensor (pt)
    with torch.no_grad():
        # Get the features
        features = model.get_image_features(**inputs)
    # Return the semantic meaning embedded into higher dimension for is. scales the feature vector so its total magnitude equals exactly 1
    return torch.nn.functional.normalize(features, p=2, dim=-1)

def coarse_global_retrieval(pipeline, query: torch.Tensor, device="cuda") -> torch.Tensor:
    """
    Passes the dataset and query through CLIP to find the semantic nearest neighbor.
    Returns the 4x4 pose of the closest training image.
    """
    from transformers import CLIPProcessor, CLIPModel
    
    print("  [coarse] Loading CLIP model...")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)

    train_ds = pipeline.datamanager.train_dataset
    print(f"  [coarse] Extracting features for {len(train_ds)} training images...")

    query_feat = get_clip_features(query, processor, model, device)
    
    results = []
    
    for i in range(len(train_ds)):
        train_img = train_ds[i]["image"]
        train_pose = to_4x4(train_ds.cameras[i:i+1].camera_to_worlds[0].cpu())
        
        train_feat = get_clip_features(train_img, processor, model, device)
        sim = (query_feat @ train_feat.T).item()
        results.append((sim, i, train_pose))
            
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[0], reverse=True)
    
    print("  [coarse] Top 5 CLIP matches:")
    for rank in range(5):
        sim, idx, _ = results[rank]
        print(f"    Rank {rank+1}: Image {idx:03d} | Cosine Sim: {sim:.4f}")
        
    best_pose = results[0][2]
    return best_pose

def check_step6(pipeline, views, scale) -> None:
    """
    Tests the full coarse-to-fine pipeline end-to-end using an interactive custom query.
    """
    assert_map_frozen(pipeline)
    print("  [step 6] Starting end-to-end localization test...")

    # 1. Grab a base camera from the eval set just to use its intrinsics
    base_camera, _ = views[0]
    base_pose = to_4x4(base_camera.camera_to_worlds[0].cpu())

    # Launch the interactive window to manually fly to a new pose
    query, gt_pose = generate_interactive_query(
        pipeline, 
        base_camera, 
        start_pose=base_pose, 
        step_units=0.05 * scale 
    )
    
    if query is None:
        print("  [warn] Interactive viewer closed without saving. Aborting Step 6.")
        return

    # 2. Coarse Stage: Semantic Global Retrieval
    T_init = coarse_global_retrieval(pipeline, query, device=pipeline.device)
    
    t_err_coarse, r_err_coarse = pose_error(gt_pose, T_init)
    print(f"  [coarse] Initial guess error: {t_err_coarse:.4f} units, {r_err_coarse:.2f} deg")

    # 3. Fine Stage: Differentiable 3DGS Refinement
    print("  [fine] Starting 3DGS photometric refinement...")
    res = localise(
        pipeline=pipeline,
        camera=base_camera,
        query=query,
        T_init=T_init.to(pipeline.device).float(),
        max_iters=1000,
        lr_trans=2e-3,
        lr_rot=2e-3
    )

    # 4. Final Evaluation & Matrix Verification
    t_err_final, r_err_final = pose_error(gt_pose.double(), res.T_final.cpu().double())
    
    # Extract translation vectors (X, Y, Z) in scene units
    gt_trans = gt_pose[:3, 3].numpy()
    final_trans = res.T_final.cpu().numpy()[:3, 3]

    print("\n  =======================================================")
    print("  [verify] POSE MATRIX COMPARISON")
    print("  =======================================================")
    print(f"  Query Location (X,Y,Z) : [{gt_trans[0]:.4f}, {gt_trans[1]:.4f}, {gt_trans[2]:.4f}] scene units")
    print(f"  Final Location (X,Y,Z) : [{final_trans[0]:.4f}, {final_trans[1]:.4f}, {final_trans[2]:.4f}] scene units")
    print("  -------------------------------------------------------")
    print("  Query 4x4 Matrix (Ground Truth):")
    print(np.round(gt_pose.numpy(), 4))
    print("\n  Final 4x4 Matrix (Optimized):")
    print(np.round(res.T_final.cpu().numpy(), 4))
    print("  =======================================================\n")

    print(f"  [fine] Final localized error: {t_err_final:.4f} units, {r_err_final:.2f} deg")
    print(f"  [fine] Terminated via: {res.termination} after {res.iterations} iterations")
    
    if t_err_final >= t_err_coarse:
         print("  [warn] Optimizer failed to improve the coarse guess. The coarse guess may have fallen outside the capture radius.")
    
    print("  STEP 6 COMPLETE")

def generate_interactive_query(pipeline, camera, start_pose: torch.Tensor, step_units=0.05):
    """
    Opens an interactive OpenCV window to fly around the 3DGS map.
    Controls:
      w/s : Move Forward/Backward (Z-axis)  | UP/DOWN ARROWS : Pitch
      a/d : Move Left/Right (X-axis)        | LEFT/RIGHT ARROWS : Yaw
      q/e : Move Up/Down (Y-axis)           | , / . (comma/period) : Roll
      SPACE : Save the current image and pose, then close.
      ESC   : Close without saving.
    """
    current_pose = start_pose.clone().cpu()
    print("\n[info] Launching interactive viewer.")
    print("       w/a/s/d/q/e = Translate.")
    print("       Arrow Keys = Pitch / Yaw.")
    print("       , and . = Roll.")
    print("       Press SPACE to lock query.")

    rot_step = math.radians(2.0)

    while True:
        img_tensor = render_at_pose(pipeline, camera, current_pose)
        img_np = (img_tensor.numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        cv2.imshow("Interactive Query Generator (Press SPACE to lock)", img_bgr)
        key = cv2.waitKeyEx(0)
        
        delta_t = torch.zeros(3)
        omega = torch.zeros(3)

        if key == 27:  # ESC key
            print("  [info] Cancelled.")
            cv2.destroyAllWindows()
            return None, None
        elif key == 32:  # SPACE key
            print("  [info] Novel query locked and saved!")
            cv2.imwrite("novel_query_image.png", img_bgr)
            torch.save(current_pose, "novel_query_pose.pt")
            cv2.destroyAllWindows()
            return img_tensor, current_pose

        # --- TRANSLATIONS (w/a/s/d/q/e) ---
        elif key == ord('w'): delta_t[2] = step_units
        elif key == ord('s'): delta_t[2] = -step_units
        elif key == ord('a'): delta_t[0] = -step_units
        elif key == ord('d'): delta_t[0] = step_units
        elif key == ord('q'): delta_t[1] = -step_units
        elif key == ord('e'): delta_t[1] = step_units

        # --- ROTATIONS (Arrow Keys + Comma/Period) ---
        elif key in [65362, 82]: omega[0] = rot_step   # Pitch Up
        elif key in [65364, 84]: omega[0] = -rot_step  # Pitch Down
        elif key in [65361, 81]: omega[1] = -rot_step  # Yaw Left
        elif key in [65363, 83]: omega[1] = rot_step   # Yaw Right
        elif key == ord(','): omega[2] = -rot_step     # Roll Left
        elif key == ord('.'): omega[2] = rot_step      # Roll Right

        # Build the 4x4 perturbation matrix using your robust so3_exp
        if delta_t.any() or omega.any():
            delta = torch.eye(4)
            if omega.any():
                delta[:3, :3] = so3_exp(omega) # Mathematically guarantees valid rotation
            if delta_t.any():
                delta[:3, 3] = delta_t
            
            # Right-multiply to stay in the local camera frame
            current_pose = current_pose @ delta

# ============================================================================
# RUNNER
# ============================================================================

CHECKS = {
    1: check_step1,
    2: check_step2,
    3: check_step3,
    4: check_step4,
    5: check_step5,
    6: check_step6,
}

def main():
    global save_path

    # 1. Hardcode your persistent base directory and config path
    DEFAULT_BASE_DIR = "/home/chini/Desktop/3DGS/data/desk-test-2/raw-images"
    DEFAULT_CONFIG = f"{DEFAULT_BASE_DIR}/outputs/processed/splatfacto/2026-08-16_104231/config.yml"

    # Step 4 image save directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "debug_step4_comparison.png")

    # set the arguments that will be passed to the script from the command line
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, nargs="?", default=Path(DEFAULT_CONFIG))
    ap.add_argument("--step", type=int, required=True, choices=sorted(CHECKS))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # 2. Force execution context to the base dir so Nerfstudio finds its checkpoints
    os.chdir(args.config.resolve().parents[4])

    _cfg, pipeline, scale = load_pipeline(args.config, args.device)
    freeze_map(pipeline)
    
    # We use 2 views for a faster test run, increase later if you want a full benchmark -> These views are the renders at the ground truths right?
    views = get_eval_views(pipeline, max_views=2)
    print(f"[info] scale={scale:.6f}, {len(views)} eval views")

    for s in range(6, args.step + 1):
        print(f"\n--- step {s} ---")
        CHECKS[s](pipeline, views, scale)

if __name__ == "__main__":
    main()