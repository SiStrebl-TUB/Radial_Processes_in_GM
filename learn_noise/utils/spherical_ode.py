import torch

# --- HELPER FUNCTIONS (Identisch wie zuvor) ---

def spherical_euler_step(y, v, h):
    """
    Führt einen Euler-Schritt AUF der Sphäre aus.
    Funktioniert auch mit negativem h (Rückwärtsintegration).
    """
    R = torch.norm(y, dim=-1, keepdim=True)
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    epsilon = 1e-8
    
    # Winkelgeschwindigkeit theta = (v / R) * h
    # h (dt) kann negativ sein -> theta wird negativ -> korrekte Richtung
    theta = (v_norm / (R + epsilon)) * h
    
    y_cos = torch.cos(theta) * y
    y_sin = torch.sin(theta) * (v / (v_norm + epsilon)) * R
    
    return y_cos + y_sin

def slerp(p0, p1, t):
    """Spherical Linear Interpolation"""
    p0_norm = p0 / (torch.norm(p0, dim=-1, keepdim=True) + 1e-8)
    p1_norm = p1 / (torch.norm(p1, dim=-1, keepdim=True) + 1e-8)
    
    dot = torch.sum(p0_norm * p1_norm, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
    
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    
    mask = sin_omega < 1e-6
    if mask.any():
        return (1.0 - t) * p0 + t * p1
        
    w0 = torch.sin((1.0 - t) * omega) / sin_omega
    w1 = torch.sin(t * omega) / sin_omega
    
    return w0 * p0 + w1 * p1

# --- DER NEUE SOLVER ---

def slerp_tvdrk3(model, y0, t_vals):
    """
    Löst die ODE auf der Sphäre für gegebene Zeitpunkte t_vals.
    
    Args:
        model: Funktion model(t, x) -> v (Erwartet t als Skalar-Tensor)
        y0: Startzustand (Batch, Dim)
        t_vals: 1D-Tensor mit Zeitpunkten [t_start, t_1, ..., t_end]
                Kann auf- oder absteigend sein.
    
    Returns:
        x_traj: (len(t_vals), Batch, Dim) - Die gesamte Trajektorie
    """
    # Trajektorie initialisieren: Startpunkt ist der erste Eintrag
    trajectory = [y0]
    
    y = y0
    device = y.device
    
    # Wir iterieren über die Intervalle zwischen den t_vals
    # Wenn t_vals Länge N hat, machen wir N-1 Schritte
    num_steps = len(t_vals) - 1
    
    for i in range(num_steps):
        # Start- und Endzeit für diesen Schritt holen
        t_curr = t_vals[i]
        t_next = t_vals[i+1]
        
        # dt berechnen (kann negativ sein, z.B. bei Generation von 1.0 -> 0.0)
        dt = t_next - t_curr
        
        # Zeit-Tensoren als Skalare erstellen (für deinen TorchWrapper!)
        t_scalar = t_curr.clone().detach().to(device).float()
        t_scalar_next = t_next.clone().detach().to(device).float()
        t_scalar_half = (t_curr + 0.5 * dt).clone().detach().to(device).float()
        
        # --- STAGE 1 ---
        # v bei t_curr
        v = model(t_scalar, y)  
        y1 = spherical_euler_step(y, v, dt)
        
        # --- STAGE 2 ---
        # v bei t_curr + dt (also t_next)
        v1 = model(t_scalar_next, y1)
        y2_temp = spherical_euler_step(y1, v1, dt)
        
        # Averaging (SLERP): y2 = 3/4 y + 1/4 y2_temp
        y2 = slerp(y, y2_temp, 0.25)
        
        # --- STAGE 3 ---
        # v bei t_curr + 0.5*dt
        v2 = model(t_scalar_half, y2)
        y3_temp = spherical_euler_step(y2, v2, dt)
        
        # Final Averaging: y_new = 1/3 y + 2/3 y3_temp
        y_next = slerp(y, y3_temp, 2.0/3.0)
        
        # Update für nächsten Schritt
        y = y_next
        
        # Ergebnis speichern
        trajectory.append(y)
        
    # Liste in Tensor umwandeln: (Time, Batch, Dim)
    return torch.stack(trajectory)

def spherical_ode_solver(ode_func, x_init, t_steps):
    x = x_init
    trajectory = [x]
    
    # 1. GANZ WICHTIG: Speichere den initialen Radius jedes Punktes!
    # Da die Bahnen normerhaltend sein sollen, ist das unser absoluter Anker.
    r_start = x_init.norm(dim=1, keepdim=True) + 1e-8
    
    for i in range(len(t_steps) - 1):
        t = torch.tensor(t_steps[i], device=x.device, dtype=torch.float32)        
        t_next = torch.tensor(t_steps[i+1], device=x.device, dtype=torch.float32)
        dt = t_next - t
        v = ode_func(t, x)
            
        # 2. Tangentiale Projektion für beliebige Radien:
        # Die korrekte Formel zieht <v, x> / ||x||^2 ab, nicht nur <v, x>
        dot_product = (v * x).sum(dim=1, keepdim=True) 
        v_tan = v - (dot_product / (r_start**2)) * x
        
        v_tan_norm = v_tan.norm(dim=1, keepdim=True) + 1e-8 
        
        # 3. Winkelberechnung angepasst an den Radius:
        # Bogenlänge s = r * theta  =>  theta = (Geschwindigkeit * Zeit) / Radius
        theta = (v_tan_norm * dt) / r_start
        
        v_tan_direction = v_tan / v_tan_norm
        
        # 4. Sphärischer Schritt auf der Kugel mit Radius r_start
        x = torch.cos(theta) * x + torch.sin(theta) * (v_tan_direction * r_start)
        
        # 5. Numerische Stabilität: Zurück auf den EIGENEN Radius zwingen, nicht auf 1.0!
        x = (x / (x.norm(dim=1, keepdim=True) + 1e-8)) * r_start
        
        trajectory.append(x)
        
    return torch.stack(trajectory)