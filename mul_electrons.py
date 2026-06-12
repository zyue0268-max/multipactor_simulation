#part 1 simulation
import numpy as np
from math import erf
import scipy.special
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import factorial
from scipy.special import gammainc, gammaincinv, gamma as gamma_func
import pandas as pd 

# part 1 this is initial energy

e_charge = 1.602e-19
me = 9.109e-31
eV_to_J = 1.602e-19
k=1.380649e-23 #J/K

#fixed  frequency
freq = 2.85e9
n_seed = 3            # from top plate
omega=2*np.pi*freq
n_cycle=4
dt_ratio=100 #300 time step per RF period
dt=1/freq/dt_ratio
n_steps=int(n_cycle*dt_ratio)

#multi mc
N_RUNS = 5     #should increase
#from standard brakdown

#define fd
fd_values=np.linspace(0.5,20,3)       #24
Vrf_V = np.linspace(30, 18000, 3)     # 51 


#part 1
#initial energy 
T=300 #k
def generate_initial_energy(T):
    r=np.random.random()
    r=max(r,1e-10)
    E_initial=-k*T*np.log(r)
    return E_initial

#velocity part
def generate_initial_particles(T,n,me,surface):
     vx_list=[]
     vy_list=[]
     vz_list=[]
     E_lsit=[]
     theta_list=[]
     phi_list=[]

     for i in range(n):
          E= generate_initial_energy(T)
          v=np.sqrt(2*E/me)
          sin_theta=np.sqrt(np.random.random())
          r_phi=np.random.random()
          theta=np.arcsin(sin_theta) # page 62 isotropy
          phi=2*np.pi*r_phi
          vx=v*np.sin(theta)*np.cos(phi)
          vy=v*np.sin(theta)*np.sin(phi)
          vz_abs=v*np.cos(theta)
          if surface=="top":
            vz=-vz_abs
          elif surface=="bottom":
            vz=vz_abs
          vx_list.append(vx)
          vy_list.append(vy)
          vz_list.append(vz)
          E_lsit.append(E)
          theta_list.append(theta)
          phi_list.append(phi)
     return (
                     np.array(vx_list),
                     np.array(vy_list),
                     np.array(vz_list),
                     np.array(E_lsit),
                     np.array(theta_list),
                     np.array(phi_list)
     )
# electric field
def E_field(t,Vrf,d,phase):
     return(Vrf/d)*np.sin(omega*t+phase)

#track time 
def track (vx0,vy0,vz0,z0,Vrf,d,phase):
            
            x,y,z=0,0,z0
            vx,vy,vz=vx0,vy0,vz0
            for step in range(n_steps):
                  t=step*dt
                  Ez=E_field(t,Vrf,d,phase)
                  az=-e_charge*Ez/me

                  vz+=az*dt
                  z+=vz*dt
                  x+=vx*dt
                  y+=vy*dt

                  if z<=0 or z>=d:
                        v_total=np.sqrt(vx**2+vy**2+vz**2)
                        cos_impact=abs(vz)/v_total
                        theta_impact=np.arccos(cos_impact)
                        E_impact=0.5*me*v_total**2 /eV_to_J
                        return t,theta_impact,E_impact   
            return None,None,None



#part 2
# Furman Pive model
def heaviside(x):  # energy e and e0-e should >0
    return 1 if x>=0 else 0
    
#this part is for backscatter，elastic
def f_bs(E,E0,theta0) : #from E to E0
        cos_theta=np.cos(theta0)

        #prefactor equ3.28
        A_bs=0.02+0.476*np.exp(-E0/60.86)
        B_bs=1.26-0.26*cos_theta**2
        C_bs=2*np.exp(-(E-E0)**2/8)

        #denominator
        D_bs = 2.0 * np.sqrt(2.0 * np.pi) * erf(E0 / (2.0 * np.sqrt(2.0)))
        step=heaviside(E)*heaviside(E0-E)
        return A_bs*B_bs*C_bs*step/D_bs
        
#furman main equ

def P_bs(E0,theta0):#integral
     if E0 < 1e-10:
          return 0.0
     result ,error= quad(f_bs, 0, E0, args=(E0, theta0))
     return result

# rediffused
def f_r(E,E0,theta0):
     cos_theta=np.cos(theta0)
     #equ.3.30
     A1=0.3
     B1=np.sqrt(E)/E0**1.5
     C1=1-np.exp(-(E0/0.041)**0.104)
     D1=1.26-0.26*cos_theta**2

     step=heaviside(E)*heaviside(E0-E)
     return A1*B1*C1*D1*step

def P_rd(E0,theta0): #integral
      result, error= quad(f_r, 0, E0, args=(E0, theta0))
      return result
    
# true secondaries
# equation 3.36
def delta_star_ts(theta0):
    cos_theta0=np.cos(theta0)
    cos_theta0_08=cos_theta0**0.8
    bracket=1-cos_theta0_08
    delta_peak=1.8848*(1+0.66*bracket)
    return delta_peak
     
#equ 3.37
def E_star(theta0):
     cos_theta0=np.cos(theta0)
     bracket=1-cos_theta0
     E_peak=276.8*(1+0.7*bracket)
     return E_peak

#equ 3.38
def D_normal(ratio):
     numerator=1.54*ratio
     denominator=0.54+ratio**1.54
     return numerator/ denominator

#equ 3.35 36 37 38 
def delta_n_ts(E0,theta0):
    delta_peak=delta_star_ts(theta0)
    E_peak=E_star(theta0)
    ratio=E0/E_peak
    D_value=D_normal(ratio)
    delta=delta_peak*D_value
    return delta

def P_n_ts(n,E0,theta0):
     delta=delta_n_ts(E0,theta0)
     delta_n=delta**n
     exp_neg_delta=np.exp(-delta)
     denominator=factorial(n)
     return delta_n*exp_neg_delta/denominator

def furman_pivi_prob(E0,theta0, n_max=10): 
# equ 3.31 32 33
     p_bs=P_bs(E0,theta0)
     p_rd=P_rd(E0,theta0)
     delta=delta_n_ts(E0,theta0) 
     P_ts=[]

 #n=0 deletl
     p_n0=(1-p_bs - p_rd)*P_n_ts(0,E0,theta0) #bs 0 rd 0 ts1
     P_ts.append(max(p_n0,0)) #p_n0>0  p_n0 else 0
 #n=1
     p_n1=(1-p_bs - p_rd)*P_n_ts(1,E0,theta0)+p_bs +p_rd #bs 1 rd 1 ts1
     P_ts.append(max(p_n1,0))
#n>=2
     for n in range(2,n_max+1):
          p_n=(1-p_bs-p_rd)*P_n_ts(n,E0,theta0) #bs 0 rd 0 ts1
          P_ts.append(max(p_n,0))
     total=sum(P_ts)
     if total>0:
           P_ts= [p / total for p in P_ts] #normalize
           return p_bs, p_rd, delta, P_ts #n from 0m to 9 , normalize


# n=1 energy and angle distribution
def sample_energy_n1(E0, theta0, r1):
     E0=abs(E0)
     p_bs,p_rd,delta,P_ts=furman_pivi_prob(E0,theta0)
#equ d4a- d4c
     p_ts_total=max(1-p_bs-p_rd,0)
     p1_ts=P_n_ts(1,E0,theta0)*p_ts_total
     delta_1=p_bs+p_rd+p1_ts

     if delta_1<1e-30: #negative
          return 0
     a_bs=p_bs/delta_1
     a_rd=p_rd/delta_1
     a_ts=p1_ts/delta_1 #n=1

#case 1 bs generate random number u from0 -1
   
     if 0<=r1<a_bs:   #backscatter elastic
          sigma_e=2 #ev
          while True:
               g=np.random.standard_normal() #N(0,1)
               E_sec=E0-sigma_e*abs(g)
               if E_sec>=0:
                    break
          
          return E_sec, "BS"
    
# case 2  rd
     elif a_bs<=r1<a_bs+a_rd:
          u1=np.random.random()
          q=0.5
          E_sec=E0*u1**(1/(1+q)) #E_sec=E0*u1**2/3
          E_sec=np.clip(E_sec,0,E0)
       
          return E_sec,"RD"

#case 3 ts
     else:
          #from furman table 2 n=1
          p1=2.5
          epsilon1=1.5
          u2=np.random.random()
          P0=gammainc(p1,E0/epsilon1)
          x=gammaincinv(p1,u2*P0)
          E_sec=epsilon1*x
          return E_sec,"TS1"
          
#part 4
#n2 to 9 energy sampling
p_n_dic={
         2:3.3,
         3:2.5,
         4: 2.5,
         5: 2.8,
         6: 1.3,
         7: 1.5,
         8: 1.5,
         9: 1.5,
         10: 1.5}
epsilon_n_dic={ 
    2: 1.75,
    3: 1.0,
    4: 3.75,
    5: 8.5,
    6: 11.5,
    7: 2.5,
    8: 3.0,
    9: 2.5,
    10: 3.0
     
}
def sample_ts_energies(E0,n):
     E0=abs(E0)
     n=int(n)
    
     if n<2:
          raise ValueError("n must be >=2")
     p_shape=p_n_dic[n]
     epsilon_n=epsilon_n_dic[n]
     
     x0=E0/epsilon_n   # should depend on n, for simplify use 1.75 for all
     P0=float(gammainc(n*p_shape,x0))
     
     theta_list=[]
     for k in range(1,n): #1  to n-1
          mu=float(p_shape*(n-k))
          nu=float(p_shape)

          u_k=float(np.random.random())
          beta_inv=scipy.special.betaincinv(mu,nu,u_k)
          theta_k=np.arcsin(np.sqrt(beta_inv))
          theta_list.append(theta_k)
          
     u=float(np.random.random())
     y=np.sqrt(gammaincinv(n*p_shape,u*P0))

     y_list=[]
     sin_product=1

     for k in range(1,n+1): #1 to n for y
        if k<n: #1 to n-1 for theta
                    theta_k=theta_list[k-1]
                    y_k=y*sin_product*np.cos(theta_k)
                    y_list.append(y_k)
                    sin_product*=np.sin(theta_k)

        else:
                    y_k=y*sin_product
                    y_list.append(y_k)
            
     y_array=np.array(y_list)
     E_list=epsilon_n*y_array**2
     return E_list

#part 3 hit information and new table
def sample_n_secondary(E0,theta0):
       p_bs,p_rd ,delta,P_ts=furman_pivi_prob(E0,theta0) #get pro
       P_ts=np.array(P_ts,dtype=float)
       P_ts=np.clip(P_ts,0,None) #remove negative 
       total=np.sum(P_ts)
       if total<=0:
            return 0
       P_ts=P_ts/total
       n_values=np.arange(len(P_ts)) # n 0123
       n_sec=np.random.choice(n_values,p=P_ts)
       return int(n_sec)

# #test(passed)
# E_test = 300.0
# theta_test = 0.0

# for i in range(10):
#     n_sec = sample_n_secondary(E_test, theta_test)
#     print("sampled n_sec =", n_sec)

#part 4 v in 3 direction for secondary electrons
#assume x and y dimension are infinite , only consider z
def sample_secondary_velocity(E_sec, hit_surface):
     E_sec=max(E_sec,0)
     
     E_sec_J=E_sec*eV_to_J #ev to j  for v
     v_sec=np.sqrt(2*E_sec_J/me)
     r_theta=np.random.random() #lambertian again
     r_phi=np.random.random()
     theta_sec=np.arcsin(np.sqrt(r_theta))
     phi_sec=2*np.pi*r_phi

     #velocity components
     vx_sec=v_sec*np.sin(theta_sec)*np.cos(phi_sec)
     vy_sece=v_sec*np.sin(theta_sec)*np.sin(phi_sec)
     vz_sec=v_sec*np.cos(theta_sec)

     #which plate hit decide direction
     if hit_surface=="bottom": #for sec emit from z=0 
          vz=vz_sec
     elif hit_surface=="top":
          vz=-vz_sec
     else:
          raise ValueError("hit_surface must be top or bottom")
     return vx_sec, vy_sece ,vz


# #tset (passed)
# E_test = 5.0  # eV sec energy

# vx_sec, vy_sec, vz_sec = sample_secondary_velocity(E_test, hit_surface="bottom")
# print("bottom emission:")
# print("vx_sec =", vx_sec, "vy_sec =", vy_sec, "vz_sec =", vz_sec)
# print("vz should be positive:", vz_sec > 0)

# vx_sec, vy_sec, vz_sec = sample_secondary_velocity(E_test, hit_surface="top")
# print("top emission:")
# print("vx_sec =", vx_sec, "vy_sec =", vy_sec, "vz_sec =", vz_sec)
# print("vz_sec should be negative:", vz_sec < 0)

#part 5 secondary particles list
def generate_secondaries_after_impact(E_impact,theta_impact,hit_surface,x_hit,y_hit,z_hit,generation,parent_weight=1):
     #impact energy using furman to generate E sec
     secondaries=[]
     n_sec=sample_n_secondary(E_impact,theta_impact)
    #n=0 no emitted electrons
     if n_sec==0:
          return secondaries
     
     #n>1 1electrons
     if n_sec==1:
        r1=np.random.random()

        E_sec,sec_type=sample_energy_n1(E_impact,theta_impact,r1)  
        vx_sec,vy_sec,vz_sec=sample_secondary_velocity(E_sec=E_sec,
                                                       hit_surface=hit_surface
        )
        secondaries.append({
             "x":x_hit,
             "y":y_hit,
             "z":z_hit,
             "vx":vx_sec,
             "vy":vy_sec,
             "vz":vz_sec,
             "generation":generation,
             "type":sec_type, #from furman line 228
             "weight":parent_weight
 })
        return secondaries

#n>2 from 2 to 10 
     if n_sec>=2:
         E_sec_list=sample_ts_energies(E_impact,n_sec)
         for E_sec in E_sec_list:
              vx_sec,vy_sec,vz_sec=sample_secondary_velocity(
                   E_sec=E_sec,
                   hit_surface=hit_surface )
              secondaries.append({
               "x":x_hit,
               "y":y_hit,
                "z":z_hit,
                "vx":vx_sec,
                "vy":vy_sec,
                "vz":vz_sec,
                "generation":generation,
                "type":"TS",
                "weight":parent_weight
            })
     return secondaries

#tset

# E_test = 0.1
# theta_test = 0.0

# secondaries = generate_secondaries_after_impact(
#     E_impact=E_test,
#     theta_impact=theta_test,
#     hit_surface="bottom",
#     x_hit=0.0,
#     y_hit=0.0,
#     z_hit=1e-9,
#     generation=1

# )

# print("number of generated secondaries =", len(secondaries))
# print(secondaries)
# E_test = 0.1
# theta_test = 0.0

# p_bs, p_rd, delta, P_ts = furman_pivi_prob(E_test, theta_test)

# print("p_bs =", p_bs)
# print("p_rd =", p_rd)
# print("delta =", delta)
# print("P_ts =", P_ts)

# for n, p in enumerate(P_ts):
#     print(f"n = {n}, probability = {p:.4f}")

#part 6 initialize particles scan different fd:
def initialize_particles(n_seed,d,surface="top"):
      vx_list,vy_list,vz_list,E_list,theta_list,phi_list=generate_initial_particles(
        T=T,
        n=n_seed,
        me=me,
        surface=surface

      )

      particles=[]
      offset=1e-9 # not form z=0 in case 
      for i in range(n_seed):
           if surface=="top":
                z0=d-offset

           elif surface=="bottom":
                z0=offset
           else:
                raise ValueError("surface must be 'top" or 'bottom')

           particles.append({
            "x": 0.0,
            "y": 0.0,
            "z": z0,
            "vx": vx_list[i],
            "vy": vy_list[i],
            "vz": vz_list[i],
            "generation": 0,
            "type": "seed",#initial seeds  not sey
            "weight":1.0

           })
      return particles
      
#test

# fd_test = 5.0
# f_GHz = freq / 1e9
# d_mm_test = fd_test / f_GHz
# d_test = d_mm_test * 1e-3

# particles = initialize_particles(
#     n_seed=5,
#     d=d_test,
#     surface="top"
# )

# print("number of initial particles =", len(particles))
# print(particles[0])
# print("z should be close to d:", particles[0]["z"], "d =", d_test) #top
# print("vz should be negative:", particles[0]["vz"] < 0)

#part 7 simulation
def simulation_plate_furman(fd, Vrf, n_seed, surface="top", max_electrons=2000, phase=None):
    f_GHz = freq / 1e9
    d_mm = fd / f_GHz
    d = d_mm * 1e-3  # m

    if phase is None:
        phase = 2 * np.pi * np.random.random()

    n_step = int(n_cycle * dt_ratio)

    particles = initialize_particles(n_seed=n_seed, d=d, surface=surface)

    N_history = []
    impact_energy_history = []
    impact_angle_history = []
    hit_surface_history = []

    for step in range(n_step):
        t = step * dt
        Ez = E_field(t, Vrf, d, phase)
        az = -e_charge * Ez / me

        next_particles = []

        for p in particles:
            x = p["x"]
            y = p["y"]
            z = p["z"]
            vx = p["vx"]
            vy = p["vy"]
            vz = p["vz"]

            # update velocity and position
            vz = vz + az * dt

            x = x + vx * dt
            y = y + vy * dt
            z = z + vz * dt

            # still inside the gap
            if 0 < z < d:
                p["x"] = x
                p["y"] = y
                p["z"] = z
                p["vx"] = vx
                p["vy"] = vy
                p["vz"] = vz

                next_particles.append(p)

            # hit one of the plates
            else:
                if z <= 0:
                    hit_surface = "bottom"
                    z_hit = 1e-9
                else:
                    hit_surface = "top"
                    z_hit = d - 1e-9

                # impact speed
                v_total = np.sqrt(vx**2 + vy**2 + vz**2)

                if v_total <= 0:
                    continue

                # impact energy
                E_impact = 0.5 * me * v_total**2 / eV_to_J

                # impact angle
                cos_impact = abs(vz) / v_total
                cos_impact = np.clip(cos_impact, 0, 1)
                theta_impact = np.arccos(cos_impact)

                impact_energy_history.append(E_impact)
                impact_angle_history.append(theta_impact)
                hit_surface_history.append(hit_surface)

                # secondary electrons after impact
                secondaries = generate_secondaries_after_impact(
                    E_impact=E_impact,
                    theta_impact=theta_impact,
                    hit_surface=hit_surface,
                    x_hit=x,
                    y_hit=y,
                    z_hit=z_hit,
                    generation=p["generation"] + 1,
                    parent_weight=p.get("weight",1)
                )

                next_particles.extend(secondaries)

        # update particles
        particles = next_particles

        # total physical electrons with macro-particle weights
        total_physical_electrons = sum(p.get("weight",1) for p in particles)
        N_history.append(total_physical_electrons)

        if total_physical_electrons == 0:
            break

        # weighted resampling instead of physical truncation
        if len(particles) > max_electrons:
            total_weight_before = sum(p["weight"] for p in particles)

            idx = np.random.choice(
                len(particles),
                size=max_electrons,
                replace=False
            )

            survivors = [particles[i] for i in idx]
            weight_after_sample = sum(p["weight"] for p in survivors)

            if weight_after_sample > 0:
                scale_factor = total_weight_before / weight_after_sample

                for p in survivors:
                    p["weight"] *= scale_factor

            particles = survivors

    final_physical_N = sum(p["weight"] for p in particles)

    return {
        "fd": fd,
        "d_mm": d_mm,
        "d_m": d,
        "Vrf": Vrf,
        "phase": phase,
        "N_history": np.array(N_history),
        "impact_energy_eV": np.array(impact_energy_history),
        "impact_angle_rad": np.array(impact_angle_history),
        "hit_surface": hit_surface_history,
        "final_N": final_physical_N,
        "n_macro_particles": len(particles),
        "n_impacts": len(impact_energy_history)
    }

#test

# result = simulation_plate_furman(
#     fd=5.0,
#     Vrf=1000.0,
#     n_seed=5,
#     surface="top",
#     max_electrons=2000,
#     phase=0.0
# )

# print("final_N =", result["final_N"])
# print("n_impacts =", result["n_impacts"])
# print("impact energy samples =", result["impact_energy_eV"][:10])
# print("impact angle samples deg =", np.degrees(result["impact_angle_rad"][:10]))

# plt.figure(figsize=(7, 4))
# plt.plot(result["N_history"])
# plt.xlabel("time (RF cycles)")
# plt.ylabel("number of electrons")
# plt.title("Electron number history")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()



#part 8
def check_multipactor_R2(N_history, dt_ratio, n_cycle):
    N_history = np.array(N_history, dtype=float)

    # 数据太少
    if len(N_history) < dt_ratio:
        return False, 0.0, 0.0, np.array([])

    envelope_N = []

    # 有多少个完整 RF cycle
    max_cycle_available = min(n_cycle, len(N_history) // dt_ratio)

    for i in range(max_cycle_available):
        start = i * dt_ratio
        end = (i + 1) * dt_ratio

        cycle_N = N_history[start:end]

        if len(cycle_N) == 0:
            continue

        # 每个 RF 周期取平均电子数
        envelope_N.append(np.mean(cycle_N))

    envelope_N = np.array(envelope_N, dtype=float)

    # 平均包络线点太少
    if len(envelope_N) < 3:
        return False, 0.0, 0.0, envelope_N

    # 不能对 0 或负数取 log
    if np.any(envelope_N <= 0):
        return False, 0.0, 0.0, envelope_N

    # 先判断整体有没有增长
    if envelope_N[-1] <= envelope_N[0]:
        return False, 0.0, 0.0, envelope_N

    # ln(N) = alpha * t + b
    y = np.log(envelope_N)
    t = np.arange(1, len(envelope_N) + 1)

    slope, intercept = np.polyfit(t, y, 1)
    alpha = slope

    # alpha <= 0 不算增长
    if alpha <= 0:
        return False, alpha, 0.0, envelope_N

    # 计算 R^2
    y_pred = slope * t + intercept

    SSE = np.sum((y - y_pred) ** 2)
    SST = np.sum((y - np.mean(y)) ** 2)

    if SST == 0:
        R_square = 0.0
    else:
        R_square = 1 - SSE / SST

    # 不用 0.8 阈值，这里只根据 alpha > 0 判定
    multipacting = True

    return multipacting, alpha, R_square, envelope_N


# part 9 parameter sweep with R2 criterion

total_cases = len(fd_values) * len(Vrf_V) * N_RUNS
case_id = 0
results = []

for fd in fd_values:
    for Vrf in Vrf_V:

        multi_list = []
        alpha_list = []
        R2_list = []
        final_N_list = []
        impact_energy_mean_list = []

        for run in range(N_RUNS):
            case_id += 1

            print(
                f"Running {case_id}/{total_cases}: "
                f"fd={fd:.2f}, Vrf={Vrf:.1f}, run={run + 1}"
            )

            sim_result = simulation_plate_furman(
                fd=fd,
                Vrf=Vrf,
                n_seed=n_seed,
                surface="top",
                max_electrons=2000,
                phase=None
            )

            is_multi, alpha, R2, envelope_N = check_multipactor_R2(
                sim_result["N_history"],
                dt_ratio=dt_ratio,
                n_cycle=n_cycle
            )

            multi_list.append(is_multi)
            alpha_list.append(alpha)
            R2_list.append(R2)
            final_N_list.append(sim_result["final_N"])

            if len(sim_result["impact_energy_eV"]) > 0:
                impact_energy_mean_list.append(np.mean(sim_result["impact_energy_eV"]))
            else:
                impact_energy_mean_list.append(0.0)

            print(
                f"Done {case_id}/{total_cases}: "
                f"multi={is_multi}, alpha={alpha:.4f}, R2={R2:.4f}, "
                f"final_N={sim_result['final_N']:.2f}"
            )

        results.append({
            "fd": fd,
            "Vrf": Vrf,
            "multipactor_ratio": np.mean(multi_list),
            "avg_alpha": np.mean(alpha_list),
            "avg_R2": np.mean(R2_list),
            "avg_final_N": np.mean(final_N_list),
            "avg_impact_energy_eV": np.mean(impact_energy_mean_list)
        })

df_results = pd.DataFrame(results)

print(df_results)

#test

def test_SEY_curve_basic():
    E_range = np.linspace(1, 1000, 300)
    theta0 = 0.0

    delta_list = np.array([delta_n_ts(E, theta0) for E in E_range])

    E_peak_numeric = E_range[np.argmax(delta_list)]
    delta_max_numeric = np.max(delta_list)

    print("SEY curve test:")
    print("numeric peak energy =", E_peak_numeric)
    print("numeric max delta =", delta_max_numeric)

    assert delta_max_numeric > 1.0
    assert 100 < E_peak_numeric < 600

    print("SEY curve basic test passed")

