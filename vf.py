import numpy as np

def intersect_line(X0, Y0, VX, VY, x1, y1, x2, y2):
    # 배열화
    X0 = np.asarray(X0);  Y0 = np.asarray(Y0)
    VX = np.asarray(VX);  VY = np.asarray(VY)
    dx = x2 - x1
    dy = y2 - y1
    # 행렬식(det) 계산 (선분과 광선 방향 벡터의 평행 검사)
    det = VY * dx - VX * dy
    mask_det = np.abs(det) > 1e-12  # det가 0에 가까우면 평행으로 간주
    # 출력 배열 준비 (초기값 np.inf로 설정하여 교차 없음 표시)
    t = np.full(X0.shape, np.inf)
    if not mask_det.any():
        return t  # 모든 광선이 선분과 평행인 경우
    idx = mask_det  # 유효한 인덱스 마스크
    # 광선-선분 교차 지점 계산 (해당 광선만 계산)
    X1_X0 = x1 - X0[idx]
    Y1_Y0 = y1 - Y0[idx]
    # t = (-(x1-X0)*dy + dx*(y1-Y0)) / det,   u = (VX*(y1-Y0) - VY*(x1-X0)) / det 
    t_idx = ( - X1_X0 * dy + dx * Y1_Y0 ) / det[idx]
    u_idx = ( VX[idx] * Y1_Y0 - VY[idx] * X1_X0 ) / det[idx]
    # 조건: t > 0 (광선 앞방향) 그리고 0 <= u <= 1 (선분 범위 내)
    valid = (t_idx > 1e-12) & (u_idx >= -1e-12) & (u_idx <= 1 + 1e-12)
    # 유효한 광선들에 대해서만 t 값을 업데이트, 나머지는 여전히 np.inf
    t[idx] = np.where(valid, t_idx, np.inf)
    return t

def intersect_circle(X0, Y0, VX, VY, xc, yc, r):
    X0 = np.asarray(X0);  Y0 = np.asarray(Y0)
    VX = np.asarray(VX);  VY = np.asarray(VY)
    if r <= 0:
        # 반지름이 0이면 교차 없음 (np.inf 반환)
        return np.full(X0.shape, np.inf)
    # 2차 방정식 계수 계산: A*t^2 + B*t + C = 0 
    A = VX**2 + VY**2  # (단위 방향벡터인 경우 A ~ 1)
    B = 2 * (VX * (X0 - xc) + VY * (Y0 - yc))
    C = (X0 - xc)**2 + (Y0 - yc)**2 - r**2
    disc = B**2 - 4 * A * C  # 판별식
    t = np.full(X0.shape, np.inf)
    mask = disc >= 0  # 교차 가능 여부 (판별식 >= 0)
    if not mask.any():
        return t
    # 판별식 >= 0인 광선들에 대해 두 해 계산
    sqrt_disc = np.sqrt(disc[mask])
    A_sel = A[mask];  B_sel = B[mask]
    t1 = (-B_sel - sqrt_disc) / (2 * A_sel)
    t2 = (-B_sel + sqrt_disc) / (2 * A_sel)
    # 양의 해만 고려하여 가장 작은 t 선택 (광선의 첫 교차 지점)
    t_sel = np.full(t1.shape, np.inf)
    # 두 해가 모두 양수이면 그 중 작은 값 채택
    cond1 = (t1 > 1e-12) & (t2 > 1e-12)
    t_sel[cond1] = np.minimum(t1[cond1], t2[cond1])
    # 한 해만 양수이면 그 값 채택
    cond2 = (t1 > 1e-12) & (t2 <= 1e-12)
    t_sel[cond2] = t1[cond2]
    cond3 = (t2 > 1e-12) & (t1 <= 1e-12)
    t_sel[cond3] = t2[cond3]
    # (양의 해가 없으면 np.inf 유지)
    t[mask] = t_sel
    return t

def VFRayTracing(H, W, a, ht, d, Person, MCSampleSize, NRays):

    # 지오메트리 정규화 (폭 W로 나눠서 무차원 좌표계로 처리)
    h = H / W
    w = W / W  # = 1.0
    # 나무(원형 수관) 위치 및 반지름
    xc  = 1.0 + d * w
    yc  = ht * w
    r   = a * w
    xc2 = 1.0 + w - d * w
    # 사람(점) 위치 및 반지름
    xcp6 = 1.0 + (Person.PositionPx / W) * w
    ycp6 = (Person.PositionPz / W) * w
    rp6  = 1.0 / 1000.0  # 매우 작은 반지름으로 점 표현
    # 발사 시작점의 미소 오프셋 (자기 교차 방지)
    stc = 1e-10

    # 대칭적인 광선 각도 분포 생성 (총 NRays개 각도)
    def generate_ray_angles(N):
        # 표면 법선 기준으로 대칭 분포된 고도각 배열 생성
        S = N // 2
        if N % 2 == 1:
            # NRays가 홀수: 수직 방향(90°)을 포함
            offsets = np.arcsin(np.arange(1, S+1) / float(S))
            angles = [np.pi/2]  # 수직(90°) 포함
            for off in offsets:
                angles.append(np.pi/2 - off)  # 수직보다 -off (한쪽 방향)
                angles.append(np.pi/2 + off)  # 수직보다 +off (대칭 방향)
            angles = np.array(angles)
        else:
            # NRays가 짝수: 수직 방향 제외하고 좌우 대칭 쌍으로 구성
            offsets = np.arcsin(np.arange(1, S+1) / float(S))
            angles = []
            for off in offsets:
                angles.append(np.pi/2 - off)
                angles.append(np.pi/2 + off)
            angles = np.array(angles)
        angles.sort()  # 각도를 오름차순 정렬 (낮은 고도 -> 높은 고도)
        return angles

    RayAngles = generate_ray_angles(NRays)  # 길이 NRays인 배열
    
    # 표면별 결과 저장 배열 (7개의 발신 면, 6개의 수신 대상)
    View_factor = np.zeros((7, 6))
    total_rays = MCSampleSize * NRays  # 발사되는 총 광선 수 (표면당)
    
    # 대상 면의 선분 정의 (정규화 좌표계 기준)
    ground_seg = (1.0, 0.0, 1.0 + w, 0.0)      # 지면 선분 (x:1~1+w, y=0)
    wall1_seg  = (1.0, 0.0, 1.0, h)            # 왼쪽 벽 (x=1, y:0~h)
    wall2_seg  = (1.0 + w, 0.0, 1.0 + w, h)    # 오른쪽 벽 (x=1+w, y:0~h)
    
    for surface in range(1, 8):
        # 발신 면적에서의 무작위 표본 점 생성
        RandSZ = np.random.rand(MCSampleSize)  # [0,1) 균등 난수
        if surface == 1:  # Wall 1 (왼쪽 벽)
            YSv = h * RandSZ
            XSv = np.full_like(YSv, 1.0 + stc)  # x≈1.0에서 약간 내부로 이동
            # 벽 표면은 법선이 수평(오른쪽)을 향하므로, 광선 각도를 -90°만큼 회전
            # (RayAngles는 수평면 기준 고도각이므로 -π/2로 변환하여 벽 전면 hemisphere 커버)
            angle_matrix = (RayAngles - np.pi/2)[None, :]  # (1, NRays) 행렬
            angle_matrix = np.repeat(angle_matrix, MCSampleSize, axis=0)  # (MCSampleSize, NRays)
        elif surface == 2:  # Wall 2 (오른쪽 벽)
            YSv = h * RandSZ
            XSv = np.full_like(YSv, 1.0 + w - stc)
            # 벽2는 법선이 수평 왼쪽을 향하므로 +90° 회전
            angle_matrix = (RayAngles + np.pi/2)[None, :]
            angle_matrix = np.repeat(angle_matrix, MCSampleSize, axis=0)
        elif surface == 3:  # Ground (지면)
            XSv = 1.0 + w * RandSZ
            YSv = np.full_like(XSv, stc)  # y≈0에서 조금 위로 이동
            # 지면은 위쪽 반공간으로 광선 발사 -> RayAngles 자체 사용 (0°=지평선 방향)
            angle_matrix = np.repeat(RayAngles[None, :], MCSampleSize, axis=0)
        elif surface == 4:  # Tree 1 (나무 1)
            ang = 2 * np.pi * RandSZ  # [0, 2π)에서 무작위 각도 (원 주변 점의 극각)
            # 원둘레 무작위 점 (수관 표면)
            XSv = xc + (r + stc) * np.cos(ang)
            YSv = yc + (r + stc) * np.sin(ang)
            # 각 점의 바깥쪽 법선 방향 = ang (극각), 광선 분포를 법선 기준으로 -90° 회전 후 각 점의 각도만큼 시프트
            # 이를 위해 RayAngles-π/2 (수직 기준 대칭 분포)를 각 점의 ang에 더함 
            angle_matrix = (RayAngles - np.pi/2) + ang[:, None]  # (MCSampleSize, NRays)
        elif surface == 5:  # Tree 2 (나무 2)
            ang = 2 * np.pi * RandSZ
            XSv = xc2 + (r + stc) * np.cos(ang)
            YSv = yc  + (r + stc) * np.sin(ang)
            angle_matrix = (RayAngles - np.pi/2) + ang[:, None]
        elif surface == 6:  # Sky (하늘, 상부 개방면)
            XSv = 1.0 + w * RandSZ
            YSv = np.full_like(XSv, h - stc)
            # 하늘은 아래쪽 반공간으로 광선 발사 -> +180° (수평면 기준 아래 방향) 회전
            angle_matrix = (RayAngles + np.pi)[None, :]
            angle_matrix = np.repeat(angle_matrix, MCSampleSize, axis=0)
        elif surface == 7:  # Point (사람 위치 점)
            ang = 2 * np.pi * RandSZ
            XSv = xcp6 + (rp6 + stc) * np.cos(ang)
            YSv = ycp6 + (rp6 + stc) * np.sin(ang)
            angle_matrix = (RayAngles - np.pi/2) + ang[:, None]   
        else:
            continue  # 해당 없음

        # 모든 샘플 점에 대한 광선 시작점과 방향 벡터 계산
        # 각 배열의 길이는 MCSampleSize * NRays로 동일합니다.
        X0 = np.repeat(XSv, NRays)
        Y0 = np.repeat(YSv, NRays)
        angles = angle_matrix.reshape(-1)  # 평탄화 (길이 = total_rays)
        VX = np.cos(angles)
        VY = np.sin(angles)
        
        # 각 대상에 대한 교차 거리 계산 (교차 없으면 np.inf 유지됨)
        t_g = intersect_line(X0, Y0, VX, VY, *ground_seg)  # 지면
        t_w1 = intersect_line(X0, Y0, VX, VY, *wall1_seg)   # 벽1
        t_w2 = intersect_line(X0, Y0, VX, VY, *wall2_seg)   # 벽2
        t_t1 = intersect_circle(X0, Y0, VX, VY, xc,  yc,  r)   # 나무1 (원)
        t_t2 = intersect_circle(X0, Y0, VX, VY, xc2, yc,  r)   # 나무2
        # '하늘'은 실제 경계가 없으므로, 다른 객체를 모두 피하면 하늘로 간주
        # 큰 거리 값을 부여하여 다른 교차가 없을 때 최소로 선택되도록 처리
        dmax = np.sqrt(h**2 + w**2) * 1.01
        t_sky = np.full(X0.shape, dmax)
        
        # 모든 교차 거리 배열을 묶어서 각 광선이 어디에 처음 부딪히는지 판정
        # (stack 순서: ground=0, wall1=1, wall2=2, tree1=3, tree2=4, sky=5)
        all_t = np.stack((t_g, t_w1, t_w2, t_t1, t_t2, t_sky))
        hit_index = np.argmin(all_t, axis=0)  # 각 광선에 대해 최소 t를 주는 대상 인덱스

        # 각 대상에 부딪힌 광선 개수 계산
        unique, counts = np.unique(hit_index, return_counts=True)
        hit_counts = dict(zip(unique.tolist(), counts.tolist()))
        # 대상 인덱스를 View_factor 행 순서에 매핑 (Wall1, Wall2, Ground, Tree1, Tree2, Sky)
        hits = [0, 0, 0, 0, 0, 0]  # [Wall1, Wall2, Ground, Tree1, Tree2, Sky]
        if 1 in hit_counts: hits[0] = hit_counts[1]  # wall1
        if 2 in hit_counts: hits[1] = hit_counts[2]  # wall2
        if 0 in hit_counts: hits[2] = hit_counts[0]  # ground
        if 3 in hit_counts: hits[3] = hit_counts[3]  # tree1
        if 4 in hit_counts: hits[4] = hit_counts[4]  # tree2
        if 5 in hit_counts: hits[5] = hit_counts[5]  # sky

        # 광선 비율로 변환 (View factor)
        hits = np.array(hits, dtype=float) / total_rays
        # 자기면에 대한 교차 제거 및 재정규화
        if surface <= 5:  # Wall1, Wall2, Ground, Tree1, Tree2만 자기 시야 제거
            self_col = None
            if surface == 1: self_col = 0  # Wall1 자신
            elif surface == 2: self_col = 1  # Wall2 자신
            elif surface == 3: self_col = 2  # Ground 자신
            elif surface == 4: self_col = 3  # Tree1 자신
            elif surface == 5: self_col = 4  # Tree2 자신
            if self_col is not None:
                hits[self_col] = 0.0
                remain = hits.sum()
                if remain > 1e-8:  # 합이 0이 아닌 경우만 (0인 경우는 원래 0이었던 것)
                    hits = hits / remain
        View_factor[surface-1, :] = hits
    return View_factor

def VFAnalytical(H, W):
    """
    Compute analytical view factors for an urban canyon without trees.
    Returns a dictionary of view factors (suffix _nT indicates no-tree case).
    """
    ratio = H / W  # canyon aspect ratio
    # Ground view factors (no trees)
    F_gs_nT = np.sqrt(1 + ratio**2) - ratio               # ground -> sky:contentReference[oaicite:0]{index=0}
    F_gt_nT = 0.0                                         # ground -> tree (no tree)
    F_gw_nT = 0.5 * (1 - F_gs_nT)                         # ground -> walls (split evenly to 2 walls):contentReference[oaicite:1]{index=1}
    # Wall view factors (no trees)
    F_ww_nT = np.sqrt(1 + (1/ratio)**2) - 1/ratio if ratio != 0 else 0.0  # wall -> opposite wall:contentReference[oaicite:2]{index=2}
    F_wt_nT = 0.0                                         # wall -> tree
    F_wg_nT = 0.5 * (1 - F_ww_nT)                         # wall -> ground:contentReference[oaicite:3]{index=3}
    F_ws_nT = 0.5 * (1 - F_ww_nT)                         # wall -> sky (remaining half):contentReference[oaicite:4]{index=4}
    # Sky view factors (no trees)
    F_sg_nT = F_gs_nT                                     # sky -> ground (reciprocal to ground->sky):contentReference[oaicite:5]{index=5}
    F_sw_nT = ratio * F_ws_nT if ratio != 0 else 0.0      # sky -> walls (area ratio scaled):contentReference[oaicite:6]{index=6}
    F_st_nT = 0.0                                         # sky -> tree
    # (Tree factors not applicable in no-tree scenario; set to 0)
    F_ts_nT = F_tw_nT = F_tt_nT = F_tg_nT = 0.0

    return {
        'F_gs_nT': F_gs_nT, 'F_gw_nT': F_gw_nT, 'F_ww_nT': F_ww_nT,
        'F_wg_nT': F_wg_nT, 'F_ws_nT': F_ws_nT, 'F_sg_nT': F_sg_nT,
        'F_sw_nT': F_sw_nT  # (Tree-related no-tree factors are all zero and omitted for brevity)
    }

def VFRayTracing(H, W, a, ht, d, Person=None, MCSampleSize=1000, NRays=200):
    import numpy as np
    # Normalize dimensions by canyon width W
    H, W = float(H), float(W)
    h = H / W             # normalized canyon height
    w = 1.0               # normalized canyon width (W/W)
    # Geometry coordinates (normalized)
    x_left_wall = 1.0                          # x-coordinate of left wall plane
    x_right_wall = 1.0 + w                     # x-coordinate of right wall plane
    z_ground = 0.0                             # z of ground
    z_top = h                                  # z of roof/sky level
    # Tree geometry (normalized)
    r = a * w                                  # tree radius (normalized by W)
    if r > 0:
        xc1 = 1.0 + d * w                      # center of tree1 (distance d*W from left wall)
        xc2 = 1.0 + w - d * w                  # center of tree2 (distance d*W from right wall)
        yc = ht * w                            # center height of both trees (normalized)
    else:
        xc1 = xc2 = yc = 0.0
    # Person/point geometry (normalized)
    if Person is not None:
        px = Person.get("PositionPx", 0.0) / W
        pz = Person.get("PositionPz", 0.0) / W
    else:
        px = pz = 0.0
    x_point = 1.0 + px
    y_point = pz
    rp = 1e-3                                  # small radius for point representation

    # Precompute ray altitude angle distribution (0 to pi radians)
    half = NRays // 2
    theta = np.linspace(0, 1, half + 1)        # param for altitude distribution
    angles = np.arcsin(theta)                  # altitude angles in [0, pi/2]
    # Construct symmetric set of ray angles for a full 180° hemisphere:contentReference[oaicite:8]{index=8}
    RayAngles = np.concatenate((np.pi/2 - angles[:-1][::-1], np.pi/2 + angles))  
    # Helper functions for line and circle intersection
    def intersect_line(x0, y0, vx, vy, x1, y1, x2, y2):
        """Compute parametric intersection t of rays (x0,y0)+t*(vx,vy) with line segment (x1,y1)-(x2,y2)."""
        dx, dy = (x2 - x1), (y2 - y1)
        denom = vx * dy - vy * dx
        # Parallel or no intersection if denom ~ 0
        t = np.full(x0.shape, np.inf)
        u = np.full(x0.shape, np.inf)
        valid = np.abs(denom) > 1e-12
        if np.any(valid):
            t_temp = (dx*(y0 - y1) - dy*(x0 - x1)) / denom
            u_temp = (vx*(y0 - y1) - vy*(x0 - x1)) / denom
            # Intersection if 0<=u<=1 on segment and t>0 on ray
            mask = (valid & (t_temp > 1e-12) & (u_temp >= 0.0) & (u_temp <= 1.0))
            t[mask] = t_temp[mask]
        return t

    def intersect_circle(x0, y0, vx, vy, xc, yc, r):
        """Compute parametric intersection t of rays with circle centered at (xc,yc) of radius r."""
        # Coefficients of quadratic equation: A*t^2 + B*t + C = 0
        A = vx**2 + vy**2
        B = 2 * (vx*(x0 - xc) + vy*(y0 - yc))
        C = (x0 - xc)**2 + (y0 - yc)**2 - r**2
        disc = B**2 - 4*A*C
        t = np.full(x0.shape, np.inf)
        mask = disc >= 0
        if np.any(mask):
            sqrt_disc = np.sqrt(disc[mask])
            t1 = (-B[mask] - sqrt_disc) / (2*A)
            t2 = (-B[mask] + sqrt_disc) / (2*A)
            # Take the smallest positive t (first intersection)
            t_candidates = np.stack([t1, t2])
            t_min = np.min(np.where(t_candidates > 1e-12, t_candidates, np.inf), axis=0)
            t[mask] = t_min
        return t

    # Initialize raw view factor count matrix
    VF_counts = np.zeros((7, 6))
    # Emit rays from each surface:
    # Define a helper to accumulate ray hits for a given emitter surface
    def emit_and_accumulate(emitter_index, sample_points, normal_angle):
        # sample_points: array of (x,y) emitter positions (shape: Mx2)
        # normal_angle: either a constant or array of per-sample surface normal angles (radians)
        M = sample_points.shape[0]
        K = RayAngles.size
        # Compute ray direction unit vectors for all sample points
        if np.isscalar(normal_angle):
            # Single orientation (planar surface)
            phi = normal_angle
            dir_angles = RayAngles + (phi - np.pi/2)      # rotate base distribution by (phi - 90°)
            vx = np.cos(dir_angles); vy = np.sin(dir_angles)
            # Repeat sample points and directions
            X0 = np.repeat(sample_points[:, 0], K)
            Y0 = np.repeat(sample_points[:, 1], K)
            VX = np.tile(vx, M); VY = np.tile(vy, M)
        else:
            # Varying orientation (circular surface): normal_angle is array of length M
            phi_offsets = normal_angle - np.pi/2           # array of angle offsets per sample
            # Create directions for each sample by adding its offset to all base angles
            dir_matrix = phi_offsets[:, None] + RayAngles  # shape: M x K
            VX = np.cos(dir_matrix).ravel()
            VY = np.sin(dir_matrix).ravel()
            X0 = np.repeat(sample_points[:, 0], K)
            Y0 = np.repeat(sample_points[:, 1], K)
        # Compute intersections with all target surfaces
        # Target indices: 0=wall1,1=wall2,2=ground,3=tree1,4=tree2,5=sky
        t_hits = []
        # Wall 1 segment (vertical line at x=1)
        t_hits.append(intersect_line(X0, Y0, VX, VY, x_left_wall, z_ground, x_left_wall, z_top))
        # Wall 2 segment (vertical line at x=1+w)
        t_hits.append(intersect_line(X0, Y0, VX, VY, x_right_wall, z_ground, x_right_wall, z_top))
        # Ground segment (horizontal line between walls at z=0)
        t_hits.append(intersect_line(X0, Y0, VX, VY, 1.0, z_ground, 1.0+w, z_ground))
        # Tree 1 circle
        if r > 1e-8:
            t_hits.append(intersect_circle(X0, Y0, VX, VY, xc1, yc, r))
        else:
            t_hits.append(np.full(X0.shape, np.inf))
        # Tree 2 circle
        if r > 1e-8:
            t_hits.append(intersect_circle(X0, Y0, VX, VY, xc2, yc, r))
        else:
            t_hits.append(np.full(X0.shape, np.inf))
        # Sky segment (horizontal line at z=h between walls)
        t_hits.append(intersect_line(X0, Y0, VX, VY, 1.0, z_top, 1.0+w, z_top))
        t_hits = np.stack(t_hits, axis=1)  # shape: (M*K) x 6
        # Determine which surface is hit first for each ray
        t_min = np.min(t_hits, axis=1)
        first_hit = np.argmin(t_hits, axis=1)
        # Count hits for valid rays (ignore rays that had no intersection in domain)
        # (Rays that escape the defined geometry count as "no hit" – these will be handled via reciprocity adjustments.)
        hit_mask = np.isfinite(t_min)
        hits = first_hit[hit_mask]
        # Accumulate counts
        for target_idx, count in zip(*np.unique(hits, return_counts=True)):
            VF_counts[emitter_index, target_idx] += count

    # 0: Wall 1 (left wall) – normal facing East (0 rad)
    Yw = np.random.rand(MCSampleSize) * h
    Xw = np.full(MCSampleSize, 1.0 + 1e-10)   # just inside the wall
    emit_and_accumulate(0, np.column_stack((Xw, Yw)), normal_angle=0.0)
    # 1: Wall 2 (right wall) – normal facing West (π rad)
    Yw = np.random.rand(MCSampleSize) * h
    Xw = np.full(MCSampleSize, 1.0 + w - 1e-10)
    emit_and_accumulate(1, np.column_stack((Xw, Yw)), normal_angle=np.pi)
    # 2: Ground – normal facing upward (π/2 rad)
    Xg = 1.0 + np.random.rand(MCSampleSize) * w
    Zg = np.full(MCSampleSize, z_ground + 1e-10)
    emit_and_accumulate(2, np.column_stack((Xg, Zg)), normal_angle=np.pi/2)
    # 3: Tree 1 – surface points around circle (varying normals)
    if r > 1e-8:
        angles = 2 * np.pi * np.random.rand(MCSampleSize)
        Xt = xc1 + (r + 1e-10) * np.cos(angles)
        Zt = yc  + (r + 1e-10) * np.sin(angles)
        emit_and_accumulate(3, np.column_stack((Xt, Zt)), normal_angle=angles)
    # 4: Tree 2 – similarly
    if r > 1e-8:
        angles = 2 * np.pi * np.random.rand(MCSampleSize)
        Xt = xc2 + (r + 1e-10) * np.cos(angles)
        Zt = yc  + (r + 1e-10) * np.sin(angles)
        emit_and_accumulate(4, np.column_stack((Xt, Zt)), normal_angle=angles)
    # 5: Sky – normal facing downward (3π/2 rad)
    Xs = 1.0 + np.random.rand(MCSampleSize) * w
    Zs = np.full(MCSampleSize, z_top - 1e-10)
    emit_and_accumulate(5, np.column_stack((Xs, Zs)), normal_angle=3*np.pi/2)
    # 6: Point/Person – small circle around the point location (if provided)
    angles = 2 * np.pi * np.random.rand(MCSampleSize)
    Xp = x_point + (rp + 1e-10) * np.cos(angles)
    Zp = y_point + (rp + 1e-10) * np.sin(angles)
    emit_and_accumulate(6, np.column_stack((Xp, Zp)), normal_angle=angles)

    # Convert counts to fractions (divide by total rays per emitter)
    total_rays = MCSampleSize * RayAngles.size
    VF_raw = VF_counts / float(total_rays)
    return VF_raw

def VFRayTracingReciprocity(H, W, a, ht, d, Person=None, MCSampleSize=1000, NRays=200):
    """
    Use ray tracing to compute reciprocal view factors for canyon with trees.
    Returns a tuple (ViewFactor, ViewFactorPoint):
      - ViewFactor: dict of view factors including keys with '_T' (with-tree) and '_nT' (no-tree).
      - ViewFactorPoint: dict of view factors from a point (Person) to various surfaces.
    """
    # Get raw view factor matrix from ray tracing
    VF_raw = VFRayTracing(H, W, a, ht, d, Person, MCSampleSize, NRays)
    # Self-view elimination: rescale each surface's factors to remove "self" portion:contentReference[oaicite:11]{index=11}:contentReference[oaicite:12]{index=12}
    VF = VF_raw.copy()
    for i in range(6):  # only surfaces 0-5 (walls, ground, trees, sky)
        if VF[i, i] < 0.999999:
            VF[i, :] /= (1 - VF[i, i])
        VF[i, i] = 0.0
    # Apply reciprocity adjustments:
    h = H / float(W)  # aspect ratio
    w = 1.0          # normalized width
    ViewFactor = {}
    if a <= 1e-8:
        # **No-tree case:** use simpler analytical-like adjustment
        F_gs_T = VF[2, 5]                            # ground -> sky (from raw)
        F_gt_T = 0.0
        F_gw_T = 0.5 * (1 - F_gs_T)                  # ground -> walls (split equally)
        # Sky (reciprocal to ground)
        F_sg_T = F_gs_T * (w / w)
        F_st_T = 0.0
        F_sw_T = 0.5 * (1 - F_sg_T)                  # sky -> walls
        # Wall (ensure wall sums to 1)
        F_wg_T = F_gw_T * (w / h)
        F_ws_T = F_sw_T * (w / h)
        F_ww_T = 1 - F_wg_T - F_ws_T                 # wall -> wall (remaining)
        F_wt_T = 0.0
        # Tree factors zero (no trees)
        F_tg_T = F_ts_T = F_tw_T = F_tt_T = 0.0
    else:
        # **With-tree case:** incorporate tree surfaces:contentReference[oaicite:16]{index=16}:contentReference[oaicite:17]{index=17}
        # Use raw ground and sky view factors as baseline
        F_gs_T = VF[2, 5]                            # ground -> sky (raw)
        F_gt_T = VF[2, 3] + VF[2, 4]                 # ground -> trees (sum of both trees)
        F_gw_T = 0.5 * (1 - F_gs_T - F_gt_T)         # ground -> walls (split equally)
        if F_gw_T < 0: F_gw_T = 0.0  # clamp to 0
        # Sky
        F_sg_T = VF[5, 2]                            # sky -> ground
        F_st_T = VF[5, 3] + VF[5, 4]                 # sky -> trees
        F_sw_T = 0.5 * (1 - F_sg_T - F_st_T)         # sky -> walls
        if F_sw_T < 0: F_sw_T = 0.0
        # Wall (using left wall as representative):contentReference[oaicite:18]{index=18}:contentReference[oaicite:19]{index=19}
        F_wt_T = VF[0, 3] + VF[0, 4]                 # wall -> trees (both trees)
        F_wg_T = F_gw_T * (w / h)                    # wall -> ground (by area ratio)
        F_ws_T = F_sw_T * (w / h)                    # wall -> sky (by area ratio)
        F_ww_T = 1 - F_wg_T - F_ws_T - F_wt_T        # wall -> opposite wall
        if F_ww_T < 0: F_ww_T = 0.0
        # Tree reciprocity: treat each tree surface area as A_t = 2πr (cylinder assumption)
        r = a * 1.0
        A_t = 2 * np.pi * r if r > 1e-8 else 1.0
        # Ground <-> Tree
        F_tg_T = (F_gt_T / 2.0) * (w / A_t)          # tree -> ground (each tree)
        # Sky <-> Tree
        F_ts_T = (F_st_T / 2.0) * (w / A_t)          # tree -> sky (each tree)
        # Wall <-> Tree
        F_tw_T = (h / A_t) * (F_wt_T / 2.0)          # tree -> wall (each tree)
        # Tree <-> Tree (between the two trees)
        F_tt_T = 1 - (F_tg_T + F_ts_T + 2 * F_tw_T)  # tree1 -> tree2
        if F_tt_T < 0: F_tt_T = 0.0
    # Combine with analytical no-tree results for output
    VF_no_tree = VFAnalytical(H, W)
    ViewFactor.update(VF_no_tree)  # include _nT keys
    # Add with-tree results to dictionary (suffix _T)
    ViewFactor.update({
        'F_gs_T': F_gs_T, 'F_gt_T': F_gt_T, 'F_gw_T': F_gw_T,
        'F_ww_T': F_ww_T, 'F_wt_T': F_wt_T, 'F_wg_T': F_wg_T, 'F_ws_T': F_ws_T,
        'F_sg_T': F_sg_T, 'F_sw_T': F_sw_T, 'F_st_T': F_st_T,
        'F_tg_T': F_tg_T, 'F_ts_T': F_ts_T, 'F_tw_T': F_tw_T, 'F_tt_T': F_tt_T
    })
    # Compute point view factors from raw (point is emitter index 6)
    F_pg = VF_raw[6, 2]         # point -> ground
    F_ps = VF_raw[6, 5]         # point -> sky
    F_pw1 = VF_raw[6, 0]        # point -> wall1 (left wall)
    F_pw2 = VF_raw[6, 1]        # point -> wall2 (right wall)
    F_pt  = VF_raw[6, 3] + VF_raw[6, 4]   # point -> trees (sum of both)
    ViewFactorPoint = {
        'F_pg': F_pg, 'F_ps': F_ps, 'F_pt': F_pt,
        'F_pwLeft': F_pw1, 'F_pwRight': F_pw2
    }
    return ViewFactor, ViewFactorPoint

M = 1000
K = 200

def fix_ray_angles(N):
    return np.linspace(0, np.pi, N)  # 간단한 버전 사용

RayAngles = fix_ray_angles(K)        # 길이 정확히 K

angle_matrix = np.repeat(RayAngles[None, :], M, axis=0)  # shape: (M, K)
angles_flat = angle_matrix.reshape(-1)                   # shape: (M*K,)

sample_points = np.random.rand(M, 2)
X0 = np.repeat(sample_points[:, 0], K)
Y0 = np.repeat(sample_points[:, 1], K)

VX = np.cos(angles_flat)
VY = np.sin(angles_flat)

print("X0", X0.shape)
print("VX", VX.shape)

