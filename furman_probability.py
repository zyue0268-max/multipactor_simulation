#part 1 simulation
import numpy as np
from math import erf
import scipy.special
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import factorial
from scipy.special import gammainc, gammaincinv, gamma as gamma_func
import pandas as pd 

# constants
e_charge = 1.602e-19
m_e = 9.109e-31
eV_to_J = 1.602e-19
k=1.380649e-23 #J/K

# copper SEY parameters
delta_max0 = 1.8
E_max0 = 300.0         #peak energy for copper SEY
Eth = 10.0             #threshold energy for copper
ks = 1.5               # angular correction factor, could be smaller

freq = 2.85e9
n = 5        # from top plate

# scan
N_RUNS = 5

# fd_values = np.linspace(0.7, 20, 20)   # GHz·mm (51grids in reference)
# Vrf_V = np.linspace(500, 8000, 20)     # V 24    (grids)

#initial energy 
T=300 #k
def generate_initial_energy(T):
    r=np.random.random()
    E_initial=-k*T*np.log(r)
    return E_initial

#velocity part
def generate_initial_particles(T,n,m_e):
     vx_list=[]
     vy_list=[]
     vz_list=[]
     for i in range(n):
          E= generate_initial_energy(T)
          v=np.sqrt(2*E/m_e)
          theta=np.random.uniform(0,np.pi)
          phi=np.random.uniform(0,2*np.pi)
          
          vx_list.append(v*np.sin(theta)*np.cos(phi))
          vy_list.append(v*np.sin(theta)*np.sin(phi))
          vz_list.append(v*np.cos(theta))
     return np.array(vx_list),np.array(vy_list),np.array(vz_list)
# impact energy
def impact_energy(vx,vy,vz,m_e,e):
     v=vx**2+vy**2+vz**2
     E_impact=0.5*m_e*v/eV_to_J
     return E_impact
vx_list, vy_list, vz_list = generate_initial_particles(T, n, m_e)

E_impact = impact_energy(vx_list, vy_list, vz_list, m_e, eV_to_J)

print(f"E_impact = {E_impact} eV")

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

#plots

# plot 1  from 0 to 9
E0_range = np.linspace(0.1, 500, 200) #no rf field E0=E initial
theta0_test = 0.0
     
p_bs_list = []
p_rd_list = []
p_total_list=[]
p_ts_dict = {i: [] for i in range(1,11)}

# with  pbr ps
for E0 in E0_range:
#       p_bs, p_rd, delta, P_ts = furman_pivi_prob(E0, theta0_test)
        p_bs = P_bs(E0, theta0_test)
        p_rd = P_rd(E0, theta0_test)

        p_ts_total = 1.0 - p_bs - p_rd
        p_ts_total = max(p_ts_total, 0.0)
        p_bs_list.append(p_bs)
        p_rd_list.append(p_rd)
        p_total=p_bs+p_rd

        for i in range(1,11):
             p_tsi=p_ts_total * P_n_ts(i, E0, theta0_test)
             p_ts_dict[i].append(p_tsi)
             p_total+=p_tsi
        p_total_list.append(p_total)
colors = plt.cm.tab20(np.linspace(0, 1, 12))

plt.figure(figsize=(12, 7))

plt.plot(E0_range, p_bs_list, color=colors[0], linewidth=2, label="p_bs")
plt.plot(E0_range, p_rd_list, color=colors[1], linewidth=2, label="p_rd")

for i in range(1, 11):
    plt.plot(
        E0_range,
        p_ts_dict[i],
        color=colors[i + 1],
        linewidth=2,
        label=f"p_ts{i}"
    )

plt.plot(E0_range, p_total_list, "k--", linewidth=2.5, label="p_total")
plt.axhline(y=1.0, color='gray', linestyle=':', linewidth=2, label='p_total = 1')

plt.xlabel("E0 (eV)")
plt.ylabel("Probability")
plt.title("Task 1: p_bs, p_rd, p_ts1-p_ts10 and p_total vs E0")
plt.grid(True, alpha=0.3)
plt.legend(ncol=2, fontsize="small")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

#plot 2 delta vs energy
delta_bs_list=p_bs_list
delta_r_list=p_rd_list
delta_ts_list=[]

delta_ts_list = []

for idx in range(len(E0_range)):
    delta_ts = 0

    for i in range(1, 11):
        delta_ts += i * p_ts_dict[i][idx]

    delta_ts_list.append(delta_ts)

delta_total_list = []

for dbs, dr, dts in zip(delta_bs_list, delta_r_list, delta_ts_list):
    delta_total_list.append(dbs + dr + dts)
plt.figure(figsize=(10,6))
plt.plot(E0_range, delta_bs_list, linewidth=2, label="delta_bs = p_bs")
plt.plot(E0_range, delta_r_list, linewidth=2, label="delta_r = p_rd")
plt.plot(E0_range, delta_ts_list, linewidth=2.5, label="delta_ts = sum(i*p_tsi)")
plt.plot(E0_range, delta_total_list, "k--", linewidth=2.5, label="delta_total")

plt.xlabel("E0 (eV)")
plt.ylabel("Delta contribution")
plt.title("Task 3: delta_bs, delta_r, and delta_ts(1-10) vs E0")
plt.grid(True, alpha=0.3)
plt.legend(fontsize="small")
plt.tight_layout()
plt.show()


#       for n in range(11):
#           P_n_dict[n].append(P_ts[n])

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

#   # : n=1 include p_bs + p_rd
# colors = plt.cm.tab10(np.linspace(0, 1, 10))
# for n in range(11):
#       ax1.plot(E0_range, P_n_dict[n], color=colors[n], linewidth=1.5,
# label=f'n={n}')
# ax1.set_xlabel('E0 (eV)')
# ax1.set_ylabel('Probability')
# ax1.set_title('Version 1: n=1 includes p_bs + p_rd')
# ax1.legend(ncol=2, fontsize='small')
# ax1.grid(True, alpha=0.3)
# ax1.set_ylim(0, 1)
# # sum of probabilities

# total_prob = np.zeros(len(E0_range))

# for n in range(11):
#     total_prob += np.array(P_n_dict[n])

# print("Min =", total_prob.min())
# print("Max =", total_prob.max())

# plot 0 to 10 ,seperate pbs prd


# from. 1 3 to 10
# for n in [0] + list(range(2, 11)):
#       ax2.plot(E0_range, P_n_dict[n], color=colors[n], linewidth=1.5,
#   label=f'n={n}')

#   # n=1  true secondary : [1-p_bs-p_rd]*P_{1,ts}
# p1_ts_list = []
# for E0 in E0_range:
#       p_bs, p_rd, delta, P_ts = furman_pivi_prob(E0, theta0_test)
#       p1_ts_only = (1.0 - p_bs - p_rd) * P_n_ts(1, E0, theta0_test)
#       p1_ts_list.append(p1_ts_only)

# ax2.plot(E0_range, p1_ts_list, color=colors[1], linewidth=1.5, label='n=1 (ts only)')
# ax2.plot(E0_range, p_bs_list, 'r--', linewidth=2, label='p_bs')
# ax2.plot(E0_range, p_rd_list, 'b--', linewidth=2, label='p_rd')

# arr_p_bs = np.array(p_bs_list)
# arr_p_rd = np.array(p_rd_list)
# arr_p_ts1 = np.array(p1_ts_list)

# p_total_array = arr_p_bs + arr_p_rd + arr_p_ts1 #n=1 only

# for n in [0]+list(range(2,11)): #n=0 and 2 to 11
#      p_total_array += np.array(P_n_dict[n]) #add 

# ax2.plot(E0_range, p_total_array, 'k-', linewidth=3, label='p_total')


# ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=2, label='p_total = 1') #1

# ax2.set_xlabel('E0 (eV)')
# ax2.set_ylabel('Probability')
# ax2.set_title('Version 2: Total Probability vs E0')
# ax2.legend(ncol=2, fontsize='small', loc='best')
# ax2.grid(True, alpha=0.3)
# ax2.set_ylim(0, 1.1)  

# plt.tight_layout()
# plt.show()



# #this part is for probabilities with different n 
# def _build_local_coords(normal):
#     normal = np.asarray(normal, dtype=float)
#     normal = normal / np.linalg.norm(normal) #unit vector

#     # 
#     if abs(normal[2]) < 0.9:
#         reference_vector = np.array([0.0, 0.0, 1.0])
#     else:
#         reference_vector = np.array([1.0, 0.0, 0.0])

#     t1 = np.cross(reference_vector, normal)
#     t1 = t1 / np.linalg.norm(t1)

#     t2 = np.cross(normal, t1)
#     t2 = t2 / np.linalg.norm(t2) #t2=normal*t1

#     return normal, t1, t2


# def _emit_velocity(normal, t1, t2, v_mag, theta_emit, phi):#z,x,y
#     return (
#         v_mag * np.cos(theta_emit) * normal
#         + v_mag * np.sin(theta_emit) * np.cos(phi) * t1 
#         + v_mag * np.sin(theta_emit) * np.sin(phi) * t2
#     )

 



# n=1 energy and angle distribution
def sample_energy_n1(E0, theta0, r1):
     E0=abs(E0)
     theta0=np.clip(theta0,0,np.pi/2)
     #normal, t1, t2=_build_local_coords(normal)
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
          #v_mag=np.sqrt(2*E_sec*eV_to_J/m_e)
          #v_vec =-np.asarray(v_in,dtype=float ) #only change direction
          return E_sec, "BS"
    
# case 2  rd
     elif a_bs<=r1<a_bs+a_rd:
          u1=np.random.random()
          q=0.5
          E_sec=E0*u1**(1/(1+q)) #E_sec=E0*u1**2/3
          E_sec=np.clip(E_sec,0,E0)
          #v_mag=np.sqrt(2*E_sec*eV_to_J/m_e)
          #theta_emit = np.arcsin(np.sqrt(np.random.random()))  # Lambert
          #phi = 2 * np.pi * np.random.random()
         # v_vec = _emit_velocity(normal, t1, t2, v_mag, theta_emit, phi)
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
        #   v_mag=np.sqrt(2*E_sec*eV_to_J/m_e)
        #   theta_emit = np.arcsin(np.sqrt(np.random.random()))  # Lambert
        #   phi = 2 * np.pi * np.random.random()
        #   v_vec = _emit_velocity(normal, t1, t2, v_mag, theta_emit, phi)
          return E_sec,"TS1"
          
#energy distribution for n>2
E0_fixed=300 #middle 

r1_list=np.linspace(0,1,1000)
r_bs,E_bs=[],[]
r_rd,E_rd=[],[]
r_ts,E_ts=[],[]
for r1 in r1_list:
     E_sec,event=sample_energy_n1(E0_fixed,theta0_test,r1)
     if event=="BS": #which range
          r_bs.append(r1)
          E_bs.append(E_sec)
     if event=="RD":
          r_rd.append(r1)
          E_rd.append(E_sec)
     elif event=="TS1":
          r_ts.append(r1)
          E_ts.append(E_sec)

p_bs = P_bs(E0_fixed, theta0_test)
p_rd = P_rd(E0_fixed, theta0_test)
p_bs_total=1-p_bs-p_rd
p_bs_total=max(p_bs_total,0)
p_ts1 = p_ts_total * P_n_ts(1, E0, theta0_test)

delta_1 = p_bs + p_rd + p_ts1

a_bs = p_bs / delta_1
a_rd = p_rd / delta_1
a_ts1 = p_ts1 / delta_1

print("a_bs =", a_bs)
print("a_r =", a_rd)
print("a_ts1 =", a_ts1)
print("a_sum =", a_bs + a_rd + a_ts1)
plt.figure(figsize=(10, 6))

plt.scatter(r_bs, E_bs, s=10, alpha=0.6, label="BS")
plt.scatter(r_rd, E_rd, s=10, alpha=0.6, label="RD")
plt.scatter(r_ts, E_ts, s=10, alpha=0.6, label="TS1")

# boundaries
plt.axvline(x=a_bs, linestyle="--", linewidth=2, label="a_bs boundary")
plt.axvline(x=a_bs + a_rd, linestyle="--", linewidth=2, label="a_bs + a_r boundary")

plt.xlabel("r1")
plt.ylabel("Emitted energy E (eV)")
plt.title(f"n = 1 energy sampling: E vs r1, E0 = {E0_fixed} eV")
plt.grid(True, alpha=0.3)
plt.legend(fontsize="small")
plt.tight_layout()
plt.show()



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
     
# plot
E0_fixed = 300
N_samples = 500

plt.figure(figsize=(10, 6))

colors = plt.cm.tab10(np.linspace(0, 1, 9))

for i, n in enumerate(range(2, 11)):
    E_all = []
    n_all = []

    for sample in range(N_samples):
        E_list = sample_ts_energies(E0_fixed, n)

        for E_sec in E_list:
            E_all.append(E_sec)
            n_all.append(n)

    x_values = np.array(n_all) + 0.12 * (np.random.random(len(n_all)) - 0.5)

    plt.scatter(
        x_values,
        E_all,
        s=8,
        alpha=0.4,
        color=colors[i],
        label=f"TS n={n}"
    )

plt.xlabel("Number of true secondary electrons n")
plt.ylabel("Emitted energy E_k (eV)")
plt.title(f"TS energy sampling for n = 2 to 10, E0 = {E0_fixed} eV")

plt.xticks(range(2, 11))

plt.grid(True, alpha=0.3)
plt.legend(ncol=2, fontsize="small")
plt.tight_layout()
plt.show()

#table and probability


# table mulit sampling

def table_probability(E0, theta0):
    E0 = float(abs(E0))
    theta0 = np.clip(theta0, 0, np.pi / 2)

    p_bs = P_bs(E0, theta0)
    p_rd = P_rd(E0, theta0)

    p_ts_total = 1.0 - p_bs - p_rd
    p_ts_total = max(p_ts_total, 0.0)

    pro = {}

    # TS n = 0 to 9
    for n in range(0, 11):
        pro[f"n={n}"] = p_ts_total * P_n_ts(n, E0, theta0) #weight

    # BS and RD separately
    pro["BS"] = p_bs
    pro["RD"] = p_rd

    return pro

def choose_event_from_pro(pro):
    r = np.random.random() #random value and decide which event
    cumulative = 0.0 #beginning 0

    event_order = [f"n={n}" for n in range(0, 11)] + ["BS", "RD"]

    for event in event_order:
        cumulative += pro[event]

        if r < cumulative: #r in this event
            return event

    # fallback for floating point error
    return event_order[-1] #in rd

#this part is for multi electron using sampling
# def pro_table(E0_range, theta0, N_samples=500): #each pro 500 times 
    

#     event_order = [f"n={n}" for n in range(0, 11)] + ["BS", "RD"]

#     for E0 in E0_range:
#         pro = table_probability(E0, theta0)
#         row={"E0":E0}
#         # counts start from 0
#         count_dict = {event: 0 for event in event_order}

#         # repeat sampling
#         for sample in range(N_samples):
#             event = choose_event_from_pro(pro)
#             count_dict[event] += 1 #500 in total

#         # one row for this E0
#         row = {"E0": E0}

#         # convert count to probability
#         for event in event_order:
#             row[event] = count_dict[event] / N_samples #eg:160/500

#         # probability sum check
#         row["p_sum"] = round(sum(row[event] for event in event_order),4)
#         rows.append(row)

#     df = pd.DataFrame(rows)
#     return df

def pro_table(E0_range, theta0): 
    rows = [] # empty table

    event_order = [f"n={n}" for n in range(0, 11)] + ["BS", "RD"]

    for E0 in E0_range:
        pro = table_probability(E0, theta0)

        
        row = {"E0": round(E0,4)}

        for event in event_order:
            row[event] = round(pro[event],4)

        # pro sum check
        row["check"] = round(sum(row[event] for event in event_order), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

# return table
E0_range = np.linspace(0.1, 500, 200)
theta0_test = 0.0

#pro table
df_wide_exact = pro_table(E0_range, theta0_test)
print(df_wide_exact)


# run 
E0_range = np.linspace(0.1, 500, 200)
theta0_test = 0.0
N_samples = 500

df_wide_exact = pro_table(       
    E0_range,
    theta0_test,
    
)

print(df_wide_exact)


top_header = (
    ["E0"]
    + [f"n={n}" for n in range(0, 11)]
    + ["BS", "RD", "check"]
)

bottom_header = (
    ["E0"]
    + [f"P{n}(E0)" for n in range(0, 11)]
    + ["Pbs(E0)", "Pr(E0)", "sum"]
)

df_wide_exact.columns = pd.MultiIndex.from_arrays(
    [top_header, bottom_header]
)

print(df_wide_exact)

#save
output_path = "/Users/yuezeng/Downloads/thesis/multi_electron/repeated_wide_probability_table.csv"

df_wide_exact.to_csv(output_path, index=False,float_format="%.4f")

print("Saved:", output_path)


# plot probabilities for different n
pro_300=table_probability(300,0)
n_values=np.arange(0,11)
P_n_values = [pro_300.get(f"n={i}", 0.0) for i in n_values]
P_n_values[1]+=pro_300["BS"]+pro_300["RD"] #ts and rd and bs

plt.figure(figsize=(10, 6))
plt.plot(n_values, P_n_values, 'r--x', markersize=8, linewidth=1.2, label='Cu')

plt.xlabel("Number of secondary electrons n",fontsize=12)
plt.ylabel("Probability P_n",fontsize=12)
plt.title("Probability for emitting n secondary electrons\n(E0 = 300 eV, normal incidence)", fontsize=12)

plt.xlim(-0.1, 10.1)
plt.ylim(0, 0.9)
plt.xticks(np.arange(0, 11, 1))
plt.grid(True, alpha=0.3) 

plt.legend(loc='center right', fontsize=12)
plt.tight_layout()
plt.show()