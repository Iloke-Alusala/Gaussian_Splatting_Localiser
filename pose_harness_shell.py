"""
pose_harness_shell.py
============
SKELETON. The pipeline logic is yours to write; the checks are already written
so they can grade you.

HOW TO USE THIS FILE
--------------------
Work one step at a time, in order. Each step has:

    - a stub function with its contract in the docstring and a TODO list
    - a check_stepN() function, already implemented, that tests your work
      against known answers

Run a step's check from the command line:

    python pose_harness_shell.py /path/to/config.yml --step 1

The runner stops at the first failure and tells you which assumption broke.
Do not move to step N+1 until step N passes. Step 1 is the only one that can
fail silently, so it is gated hardest.

Design rationale for every decision here is in LOCALISER_ROADMAP.md. Section
references in the TODOs point there. Module references point at TIER1_PLAN.md.

WHAT IS ALREADY DONE FOR YOU AND WHY
------------------------------------
The check functions, the guardrails, and the CLI are implemented, because they
are infrastructure rather than learning content, and because a test you wrote
yourself cannot catch a misconception you also wrote into the code.

Everything that constitutes the localiser is a stub.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import cv2
import os

# Shared debug output path for localise() and the CLI runner.
save_path = None


# Reuse from the diagnostic. Do not reimplement these; a fix in one place should
# fix both programs. See roadmap section 4.
from pose_harness import (
    POSE_THRESHOLDS,
    Scorer,
    get_eval_views,
    load_pipeline,
    make_perturbation,
    perturb_pose,
    pose_error,
    threshold_recall,
    to_4x4,
)


# ============================================================================
# GUARDRAILS  (implemented - do not modify)
# ============================================================================


def assert_map_frozen(pipeline) -> None:
    """
    The single most important assertion in the build.

    If any Gaussian parameter still requires grad, the optimiser will deform the
    MAP to match the query image rather than moving the CAMERA to the right
    place. The loss will fall smoothly, the run will look like a success, and
    the pose error will not improve.

    Roadmap section 1.5. This is called automatically before every check.
    """
    live = [
        name
        for name, p in pipeline.model.named_parameters()
        if p.requires_grad
    ]
    if live:
        raise AssertionError(
            f"{len(live)} map parameters still require grad, e.g. {live[:5]}. "
            "Freeze them before optimising, or you will optimise the map "
            "instead of the pose."
        )


def freeze_map(pipeline) -> None:
    """Set requires_grad False on every model parameter. Call once at startup."""
    for p in pipeline.model.parameters():
        p.requires_grad_(False)


# ============================================================================
# STEP 1  -  DIFFERENTIABLE RENDERING
# ============================================================================

def render_differentiable(  
    pipeline,
    camera,
    c2w_4x4: torch.Tensor,
) -> torch.Tensor:
    """
    Render using model.get_outputs (which lacks the @torch.no_grad decorator of get_outputs_for_camera).
    Why You Can't Just "Look and Guess"
    The core problem is: how do you know which direction to move without the connection?
    If you sever the connection (no_grad), the computer only gets a single scalar number: Loss = 0.42.
    A single number doesn't tell you:
    - Do I move Left or Right?
    - Do I Pitch Up or Yaw Down?
    """
    from nerfstudio.cameras.cameras import Cameras

    # Ensure c2w is 3x4 or 4x4 with batch dimension
    c2w_tensor = c2w_4x4[:3, :].unsqueeze(0) if c2w_4x4.ndim == 2 else c2w_4x4[..., :3, :]

    # Construct a new Cameras object connected to our autograd pose tensor
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

    # Call get_outputs directly to bypass the @torch.no_grad wrapper on get_outputs_for_camera
    outputs = pipeline.model.get_outputs(cam)
    return outputs["rgb"]


def photometric_loss(rendered: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """
    Scalar objective. Grayscale mean squared error.

    Implemented as the default so you are not blocked, but revisit it at step 7:
    a Huber loss on the residual is what handles occlusion and lighting
    mismatch. Roadmap section 1.4.
    """
    w = torch.tensor([0.299, 0.587, 0.114], device=rendered.device)
    r_gray = (rendered * w).sum(-1)
    q_gray = (query.to(rendered.device) * w).sum(-1)
    return ((r_gray - q_gray) ** 2).mean()


def check_step1(pipeline, views, scale) -> None:
    """
    Finite-difference gradient check. The gate for the entire build.

    Compares the analytic gradient from backward() against a central difference
    for each of the six tangent components.

    PASS: the two agree to roughly 2-3 significant figures. Exact agreement is
    not expected; rasterisation is piecewise smooth, not smooth.

    Run this to test: python3 pose_harness_shell.py /home/chini/Desktop/3DGS/data/desk-test-2/outputs/processed/splatfacto/2026-08-25_111611/config.yml --step 1
    """
    # Imagine you are a movie director looking at a monitor, trying to frame a shot. The camera is mounted on a drone. You want to adjust the framing, but you can only do it by pulling strings attached directly to the pixels on your monitor—these strings run all the way back to the drone's joysticks.

    # Step 1 is wiring those strings (the PyTorch autograd graph).

    # Normally, when a framework like nerfstudio renders an image, it snips the strings (torch.no_grad()) to save memory because it assumes you just want to look at the picture. The image looks fine, but if your optimizer tries to "pull" the pixels to fix the pose, nothing happens. Step 1 forces the renderer to keep those strings fully intact so the gradient can flow from the pixels back to the camera pose.

    # freeze the map
    assert_map_frozen(pipeline)
    camera, query = views[0]
    device = pipeline.device
    gt = to_4x4(camera.camera_to_worlds[0]).to(device).float()

    def loss_of(xi_vec: torch.Tensor) -> torch.Tensor:
        T = gt @ se3_exp_torch(xi_vec)
        img = render_differentiable(pipeline, camera, T)
        return photometric_loss(img, query)

    xi = torch.zeros(6, device=device, requires_grad=True)
    L = loss_of(xi)

    if L.grad_fn is None:
        raise AssertionError(
            "Loss has no grad_fn. The graph is broken between pose and image. "
            "Step 1 TODO items 1 and 3."
        )

    L.backward()
    if xi.grad is None:
        raise AssertionError(
            "xi.grad is None. Gradients are not reaching the pose at all."
        )
    analytic = xi.grad.detach().clone()

    if torch.allclose(analytic, torch.zeros_like(analytic)):
        raise AssertionError(
            "Analytic gradient is all zeros. Either the pose is not entering "
            "the render, or an in-place assignment severed the graph."
        )

    h = 1e-4
    numeric = torch.zeros(6, device=device)
    for i in range(6):
        p = torch.zeros(6, device=device)
        p[i] = h
        with torch.no_grad():
            lp = loss_of(p)
            lm = loss_of(-p)
        numeric[i] = (lp - lm) / (2 * h)

    print("  analytic:", np.round(analytic.cpu().numpy(), 6))
    print("  numeric :", np.round(numeric.cpu().numpy(), 6))

    # Scale-aware denominator to avoid blowing up near-zero gradient components
    denom = torch.maximum(numeric.abs(), torch.tensor(0.1 * numeric.abs().max().item(), device=device))
    rel = ((analytic - numeric).abs() / denom).max().item()
    print(f"  max relative difference: {rel:.4f}")

    if rel > 0.15:
        cos = torch.nn.functional.cosine_similarity(
            analytic.unsqueeze(0), numeric.unsqueeze(0)
        ).item()
        extra = ""
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


# ============================================================================
# STEP 2  -  THE UPDATE PARAMETERISATION
# ============================================================================

def se3_exp_torch(xi: torch.Tensor) -> torch.Tensor:
    """
    SE(3) exponential map, differentiable, in torch.

    CONTRACT
        in : xi, a 6-number-vector. Convention here is:
             Translation (3 numbers): Movement along the X, Y, and Z axes.
             Rotation (3 numbers): The axis of rotation scaled by the angle of rotation (how much you rotate around X, Y, and Z).
             Fix a convention and keep it; the harness uses the same ordering.
        out: 4x4 transform, differentiable in xi.

    You wrote this in NumPy for TIER1 self-test 2. This is the torch version,
    and the differences matter.

    TODO
      1. Rodrigues for the rotation block, as in so3_exp.
      2. The V matrix for the translation block:
             V = I + ((1-cos t)/t^2) Phi + ((t - sin t)/t^3) Phi^2
         where Phi is the skew matrix of the ROTATION VECTOR ITSELF, not of the unit axis. Building it from the unit axis without multiplying back by t is the standard error; it gives 0.0944 where the answer is 0.0900.
      3. Translation = V @ rho. Note this is the OPPOSITE choice from
         make_perturbation in the harness, which sets translation directly.
         That is deliberate. The diagnostic needed exact magnitudes for its
         x-axis; the optimiser needs a smooth unconstrained parameterisation and
         does not care that a step of length 0.1 is not a displacement of 0.1.
         Roadmap section 1.1.

    THE SMALL-ANGLE TRAP, WHICH IS DIFFERENT IN TORCH
        The formulas divide by theta, so theta = 0 needs a guard. A naive
        torch.where(theta > eps, formula, fallback) still evaluates BOTH
        branches, and the NaN from the untaken branch propagates through
        backward() and poisons your gradient with NaN even though the forward
        value looks correct.
        Options: clamp theta to a floor before dividing, or use a Taylor
        expansion for the small-angle branch so neither branch produces NaN.
        This matters because xi is reset to zero every iteration, so the
        small-angle path is taken on EVERY step of the optimisation.

    Depends on TIER1 module 2.
    """
    # convert rotation vector to rotation matrix using Rodrigues' formula.
    # Since application defualts to running on GPU, we move the tensor to CPU and work
    # the translation vector.
    rho = xi[:3]
    # The rotation vector. This rotation vector holds [theta_x, theta_y, theta_z] where the magnitude of the vector is the angle of rotation and the direction is the axis of rotation. The rotation vector is a compact representation of a rotation in 3D space.
    phi = xi[3:]

    # 1. The Small-Angle Trap Guard
    # We clamp the squared magnitude. This prevents division by zero 
    # AND prevents PyTorch from returning NaN gradients at exactly 0. Since rodrigues formula has terms which divide by theta, we need to make sure that theta is never exactly 0.
    theta_sq = torch.sum(phi ** 2)
    theta = torch.sqrt(torch.clamp(theta_sq, min=1e-12))

    # 2. Build the Skew-Symmetric Matrix (Phi) -> This allows you to do the cross product in matrix form
    # Using torch.stack prevents in-place operations that break the graph.
    # z = phi[0] * 0.0 keeps the dummy zero attached to the autograd graph.

    # note that Phi is the skew-symmetric matrix of the rotation vector phi, not the unit axis. This is important for the V matrix calculation later. 
    # Also conceptually k is the axis of rotation, and theta is the angle of rotation. The rotation vector phi = k * theta. The skew-symmetric matrix Phi is used to compute the rotation matrix R using Rodrigues' formula. k is a generalised axis in that if you only have to rotate around it theta degrees, to have the equivalent rotation of all x3 degrees around the x3 different axis

    z = phi[0] * 0.0
    # where Phi is equal to theta * 
    Phi = torch.stack([
        torch.stack([z, -phi[2], phi[1]]),
        torch.stack([phi[2], z, -phi[0]]),
        torch.stack([-phi[1], phi[0], z])
    ])
    Phi2 = Phi @ Phi
    I = torch.eye(3, device=xi.device, dtype=xi.dtype)

    # 3. Rodrigues' Formula for Rotation (R) and 4. V Matrix for Translation
    # Use Taylor series expansion near theta = 0 to avoid division by zero and eliminate graph truncation noise
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

    # 5. Assemble the final 4x4 matrix
    # Using torch.cat avoids in-place assignments like T[:3, :3] = R
    Rt = torch.cat([R, t.unsqueeze(1)], dim=1) # Combine 3x3 R and 3x1 t into 3x4
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=xi.device, dtype=xi.dtype)
    T = torch.cat([Rt, bottom], dim=0) # Add the bottom row to make it 4x4

    return T


def apply_update(T_current: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """
    Retract a tangent-space step back onto SE(3).

    CONTRACT
        T_new = T_current . exp(xi)      -- right composition, body frame

    One line. It is here as a named function so the convention is stated once
    and cannot drift between the optimiser and the sweep.

    Right rather than left, for the same reason as perturb_pose in the harness:
    the update means the same thing regardless of where the camera sits relative
    to the world origin, and the capture radius measured by the diagnostic then
    transfers directly. Roadmap section 1.1.

    TODO: one line.
    """
    return T_current @ se3_exp_torch(xi)


def check_step2(pipeline, views, scale) -> None:
    """
    Three properties, all with known answers.
      1. A zero update is the identity.
      2. A 1 cm update along the camera's own X axis moves 1 cm, rotates 0.
      3. Applying xi then -xi returns to the start.

      Longer explanation of the test:
      1. The "Do Nothing" Sanity Check ($e^0 = I$)
      The Test: You send a command saying: "Move 0 millimeters, rotate 0 degrees."
      The Expectation: The robot arm shouldn't twitch or drift by even a micron. It must stay in the exact same spot.
      In Code: If xi = [0, 0, 0, 0, 0, 0], se3_exp_torch(xi) must return the exact 4x4 Identity matrix.
      
      2. The "Straight Line" Ruler Check (1 cm on X)
      The Test: You tell the robot: "Slide your gripper exactly 1 cm along your own right gripper rail (local X-axis)."
      The Expectation: The gripper moves exactly 1.0000 cm along that rail.
      The gripper does not tilt, twist, pitch, or yaw by even 0.0001 degree.
      In Code: Applying xi = [0.01, 0, 0, 0, 0, 0] via apply_update must yield a translation error of exactly 0.010000 and a rotation error of 0.000000 degrees.

      3. The "Round-Trip Undo" Check (xi then -xi)
      The Test: You punch in a wild, complex command: "Slide 2 cm right, 1 cm down, 0.5 cm forward, while tilting 1.7 degrees pitch, 1.1 degrees yaw, and 0.5 degrees roll." Then, you immediately command the exact opposite negative numbers to reverse the maneuver.
      The Expectation: The robot must end up at the exact same physical atom where it started.
      In Code: Because we use right-composition in the body frame T_current @ exp(xi), executing step xi and then -xi must return you to T with zero drift.
      If all three pass, it proves your geometric engine (se3_exp_torch and apply_update) obeys the fundamental laws of rigid-body physics and 3D space.
    """

    camera, _ = views[0]
    T = to_4x4(camera.camera_to_worlds[0]).double()

    z = se3_exp_torch(torch.zeros(6, dtype=torch.float64))
    err = (z - torch.eye(4, dtype=torch.float64)).abs().max().item()
    print(f"  exp(0) deviation from identity: {err:.2e}")
    if err > 1e-9:
        raise AssertionError("exp(0) is not the identity. Small-angle branch.")

    xi = torch.zeros(6, dtype=torch.float64)
    xi[0] = 0.01
    T2 = apply_update(T, xi)
    t_err, r_err = pose_error(T, T2)
    print(f"  1 cm x-step -> t_err {t_err:.6f} (expect 0.010000), "
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


# ============================================================================
# STEP 3  -  ONE GRADIENT STEP
# ============================================================================


def single_gradient_step(
    pipeline, camera, query, T: torch.Tensor, lr: float
) -> tuple[torch.Tensor, float, float]:
    """
    Take exactly one step. Return (T_new, loss_before, loss_after).

    Structurally trivial, but do it as its own function before writing the loop.
    A sign error inside a hundred-iteration loop looks like a tuning problem; in
    one step it is unmistakable.

    TODO
      1. xi = zeros(6, requires_grad=True)
      2. T_candidate = apply_update(T, xi)
      3. render, loss, backward
      4. Step DOWN the gradient. Getting this sign wrong is the most common
         single-character bug in the build.
      5. Retract with the stepped xi and return the new pose.

    Roadmap section 3, step 3.
    """
    device = T.device
    dtype = T.dtype

    # 1. Instantiate fresh tangent vector with autograd enabled
    xi = torch.zeros(6, device=device, dtype=dtype, requires_grad=True)

    # 2. Retract to get candidate pose
    # this sets T__candiate to the zero update of T, which is the same as T. This is important because it allows us to compute the loss at the current pose before taking a gradient step.
    T_candidate = apply_update(T, xi)

    # 3. Render and compute loss
    img_before = render_differentiable(pipeline, camera, T_candidate)
    loss = photometric_loss(img_before, query)
    loss_before = loss.item()

    # Backpropagate gradients strictly to xi
    loss.backward()

    # 4 & 5. Step down the gradient and retract back onto SE(3)
    with torch.no_grad():
        xi_step = -lr * xi.grad
        T_new = apply_update(T, xi_step)

        # Compute evaluated post-step loss
        img_after = render_differentiable(pipeline, camera, T_new)
        loss_after = photometric_loss(img_after, query).item()

    return T_new, loss_before, loss_after


def check_step3(pipeline, views, scale) -> None:
    """
    Perturb by 5 mm, take one step.

    PASS: loss decreases AND pose error decreases.
    If loss falls but pose error rises, the map is not frozen.
    If loss rises, the update sign is inverted.
    """
    assert_map_frozen(pipeline)
    camera, query = views[0]
    # gt is the ground truth camera pose in 4x4 homogeneous coordinates
    gt = to_4x4(camera.camera_to_worlds[0])
    # gen is a random number generator with a fixed seed for reproducibility. Set to 0
    gen = torch.Generator().manual_seed(0)
    # delta is a small perturbation to the ground truth pose, generated randomly. The perturbation is scaled by 0.005 * scale, which means it will be a small perturbation in the order of millimeters.
    delta = make_perturbation(0.005 * scale, 0.0, generator=gen)
    # T0 gives the initial perturbed pose by applying the perturbation delta to the ground truth pose gt. This is the starting point for the gradient step.
    T0 = perturb_pose(gt, delta)

    # Cast both to double for pose_error to avoid dtype mismatch
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


# ============================================================================
# STEP 4  -  THE OPTIMISATION LOOP
# ============================================================================


@dataclass
class LocaliseResult:
    """
    Everything the report will need. Populate all of it from the start;
    retrofitting diagnostics later is tedious and you will need every field.
    """
    T_final: torch.Tensor
    iterations: int
    termination: str          # "loss_plateau" | "step_small" | "max_iters"
    loss_history: list[float] = field(default_factory=list)
    pose_history: list[torch.Tensor] = field(default_factory=list)

def localise(
    pipeline,
    camera,
    query: torch.Tensor,
    T_init: torch.Tensor,
    max_iters: int = 250,
    lr_trans: float = 2e-3,
    lr_rot: float = 2e-3,
    tol_loss: float = 1e-7,
    tol_step: float = 1e-6,
) -> LocaliseResult:
    """
    The localiser. Descend the photometric objective from T_init.
    """
    device = T_init.device
    dtype = T_init.dtype

    T_curr = T_init.clone()
    loss_history: list[float] = []
    pose_history: list[torch.Tensor] = [T_curr.clone()]

    termination = "max_iters"
    actual_iters = 0
    prev_loss: float | None = None

    # 1. Decoupled parameter groups for translation and rotation
    rho = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)
    phi = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)

    # 2. PyTorch Native Adam automatically handles bias correction (prevents first-step explosion)
    optimizer = torch.optim.Adam([
        {"params": [rho], "lr": lr_trans},
        {"params": [phi], "lr": lr_rot},
    ])
    
    # 3. Scheduler to smoothly decay step sizes and guarantee settling without bouncing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-6)

    first_render = None
    last_render = None

    for i in range(max_iters):
        actual_iters = i + 1
        optimizer.zero_grad()

        # Zero the tangent vector values to reset the Taylor expansion, but keep Adam's momentum intact!
        with torch.no_grad():
            rho.zero_()
            phi.zero_()

        # 4. Concatenate and retract
        xi = torch.cat([rho, phi], dim=0)
        T_candidate = apply_update(T_curr, xi)

        # 5. Forward & Loss
        rendered = render_differentiable(pipeline, camera, T_candidate)
        if i == 0:
            first_render = rendered.detach().clone()
        last_render = rendered.detach().clone()

        loss = photometric_loss(rendered, query)
        current_loss = loss.item()
        loss_history.append(current_loss)

        if prev_loss is not None and abs(prev_loss - current_loss) < tol_loss:
            termination = "loss_plateau"
            break
        prev_loss = current_loss

        # 6. Backward & Step
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 7. Apply the computed step to the accumulated pose
        with torch.no_grad():
            xi_step = torch.cat([rho, phi], dim=0)
            step_norm = torch.linalg.norm(xi_step).item()

            if i % 25 == 0 or i == max_iters - 1:
                mse = torch.nn.functional.mse_loss(rendered, query.to(device)).item()
                psnr = -10.0 * math.log10(mse) if mse > 0 else float("inf")
                print(f"    iter {i:03d} | loss: {current_loss:.6f} | psnr: {psnr:.2f} dB | |step|: {step_norm:.2e}")

            if step_norm < tol_step:
                termination = "step_small"
                break

            T_curr = apply_update(T_curr, xi_step)
            pose_history.append(T_curr.clone())

    # 8. Save Visuals
    if first_render is not None and last_render is not None:
        import os, cv2
        import numpy as np
        q_img = (query.detach().cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
        init_img = (first_render.detach().cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
        final_img = (last_render.detach().cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
        diff_init = np.abs(init_img.astype(float) - q_img.astype(float)).astype(np.uint8)
        diff_final = np.abs(final_img.astype(float) - q_img.astype(float)).astype(np.uint8)

        black = np.zeros_like(q_img)
        row1 = np.hstack([q_img, init_img, final_img])
        row2 = np.hstack([black, diff_init, diff_final])
        grid = np.vstack([row1, row2])
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, "debug_step4_comparison.png")
        cv2.imwrite(save_path, grid)
        print(f"  [debug] Saved visual comparison grid -> {save_path}")

    return LocaliseResult(
        T_final=T_curr,
        iterations=actual_iters,
        termination=termination,
        loss_history=loss_history,
        pose_history=pose_history,
    )

def check_step4(pipeline, views, scale) -> None:
    """
    Recover a known 1 cm / 1 degree perturbation.

    PASS: final translation error below 1 mm, rotation below 0.1 degrees.
    You know the truth because you applied the perturbation.
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
    t_m = t_err / scale if scale else float("nan")

    print(f"  initial loss: {res.loss_history[0]:.6f} -> final loss: {res.loss_history[-1]:.6f}")
    print(f"  terminated: {res.termination} after {res.iterations} iters")
    print(f"  final error: {t_m*1000:.2f} mm, {r_err:.4f} deg")

    if res.termination == "max_iters":
        print("  NOTE: hit the iteration cap. Not a convergence.")
    if t_m > 0.001 or r_err > 0.1:
        raise AssertionError(
            "Did not recover a small perturbation. If the final loss is low but "
            "the pose is wrong, you are in a local minimum, which should not "
            "happen at 1 cm. Re-check that the map is frozen."
        )
    print("  STEP 4 PASS")


# ============================================================================
# STEP 5  -  THE OPTIMISER'S CAPTURE RADIUS
# ============================================================================


def capture_radius_sweep(
    pipeline, views, scale, magnitudes, n_trials: int = 10
) -> list[dict]:
    """
    Success rate as a function of initial pose error. The result that closes the
    loop with the diagnostic.

    CONTRACT
        out: rows with initial error, final error, termination, success flags

    This is the harness sweep with `localise` where `score` used to be. Reuse
    make_perturbation, pose_error and threshold_recall unchanged.

    TODO
      1. For each view, magnitude, trial: perturb, localise, record initial and
         final error and the termination reason.
      2. Classify success with threshold_recall's pairs. Feed it METRES.
      3. Plot success rate against initial error.

    WHAT THE ANSWER MEANS
      Compare where this curve collapses against where the photometric curve
      went flat in the diagnostic. The optimiser's radius should be at or inside
      the objective's. Substantially smaller means the signal is there and your
      optimiser is not exploiting it, which is an optimisation problem and
      tractable. Larger means the two measurements are inconsistent and one is
      wrong.

    Roadmap section 3, step 5.
    """
    raise NotImplementedError("step 5")


# ============================================================================
# STEP 6  -  WIDENING THE BASIN
# ============================================================================


def localise_coarse_to_fine(
    pipeline, camera, query, T_init, schedule=(8.0, 4.0, 2.0, 0.0)
) -> LocaliseResult:
    """
    Anneal a blur (or a resolution pyramid) across the optimisation.

    Blurring attenuates the high spatial frequencies that create spurious local
    minima, leaving a broader basin around the true solution, at the cost of
    precision. So: start blurred to reach the right basin, then sharpen to
    locate it precisely. Roadmap section 2.4.

    TODO
      1. For each sigma in the schedule, blur BOTH the render and the query by
         the same amount, and run the inner loop to convergence.
      2. Carry the pose from one level to the next.
      3. The final level must be sigma = 0, or precision will be degraded.
      4. A resolution pyramid is the cheaper alternative, because fewer pixels
         are rendered as well as compared.

    TEST: rerun the step 5 sweep with this enabled. The success curve should
    shift right and small-error precision should be essentially unchanged. If
    the radius does not grow, the failure is not caused by high-frequency local
    minima, which is itself a finding and points at map quality or outliers.
    """
    raise NotImplementedError("step 6")


# ============================================================================
# STEP 7  -  ROBUSTNESS AND EVALUATION
# ============================================================================


def huber_loss(rendered, query, delta: float = 0.1) -> torch.Tensor:
    """
    Robust residual, for occlusions, moving objects and lighting mismatch.
    Quadratic within delta, linear outside, so outliers stop dominating.

    TODO: implement, then compare against photometric_loss on queries with
    occluders present. Roadmap section 1.4.
    """
    raise NotImplementedError("step 7")


def localise_multi_hypothesis(
    pipeline, camera, query, T_inits: list[torch.Tensor]
) -> LocaliseResult:
    """
    Run from several seeds, keep the lowest final loss.

    Cheap insurance against local minima, and it directly relaxes the accuracy
    your coarse pass has to deliver, which is a design argument worth making
    explicitly in the report.

    TODO: run localise on each seed, select by final loss, and record how often
    the best-loss hypothesis was also the best-pose one. Where those disagree,
    the objective is misranking, which is a reportable limitation.
    """
    raise NotImplementedError("step 7")


def evaluate(pipeline, views, scale, T_inits) -> dict:
    """
    Full evaluation over the held-out set.

    TODO produce:
      - threshold recall at each pair in POSE_THRESHOLDS
      - median translation error (metres) and rotation error (degrees)
      - median iterations to convergence
      - wall-clock per query   <- the number that matters for the Jetson work
      - the distribution of termination reasons

    Median errors here should be comparable to your step 4 single-pose result.
    Substantially worse means step 4 was tuned to one lucky view.
    """
    raise NotImplementedError("step 7")


# ============================================================================
# RUNNER  (implemented - do not modify)
# ============================================================================

CHECKS = {
    # step 1 checks wheteher the autograd graph is intact and the gradient flows from pixels to pose
    1: check_step1,
    # step 2 checks whether we will be able to translate and rotate the camera correctly using the exponential map
    2: check_step2,
    # step 3 checks whether a single gradient step reduces the loss and improves the pose
    3: check_step3,
    # step 4 checks whether the optimiser can recover a known perturbation
    4: check_step4,
}

def main():
    global save_path

    # 1. Hardcode your persistent base directory and config path
    DEFAULT_BASE_DIR = "/home/chini/Desktop/3DGS/data/desk-test-2/raw-images"
    DEFAULT_CONFIG = f"{DEFAULT_BASE_DIR}/outputs/processed/splatfacto/2026-08-16_104231/config.yml"

    # Step 4 image save directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "debug_step4_comparison.png")

    # 2. Force execution context to the base dir so Nerfstudio finds its checkpoints
    os.chdir(DEFAULT_BASE_DIR)

    # set the arguments that will be passed to the script from the command line
    ap = argparse.ArgumentParser()
    # The config file is now OPTIONAL (nargs="?"). If omitted, it uses the default.
    ap.add_argument("config", type=Path, nargs="?", default=Path(DEFAULT_CONFIG))
    # the step argument, which is required and must be one of the keys in CHECKS
    ap.add_argument("--step", type=int, required=True, choices=sorted(CHECKS))
    # the device argument, which is optional and defaults to "cuda". We default to CUDA because the harnes is designed to run on GPU, but user can override this to run on CPU.
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    _cfg, pipeline, scale = load_pipeline(args.config, args.device)
    freeze_map(pipeline)
    views = get_eval_views(pipeline, max_views=2)
    print(f"[info] scale={scale:.6f}, {len(views)} eval views")

    for s in range(1, args.step + 1):
        print(f"\n--- step {s} ---")
        CHECKS[s](pipeline, views, scale)

if __name__ == "__main__":
    main()