!pip install torch torchvision torchaudio

import streamlit as st
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import comb

############################################################
# 1) HELPER: Generate real locations (Uniform or Lumpy)
############################################################
def generate_real_locations(num_arms, distribution_type, space_dim):
    np.random.seed(123)  # fixed seed or remove if truly random each time
    if space_dim == "1D":
        if distribution_type == "Uniform":
            locs = np.random.uniform(0, 10, size=(num_arms, 1))
        else:  # Lumpy
            half = num_arms // 2
            remainder = num_arms - half
            cluster1 = 3.0 + 0.5 * np.random.randn(half)
            cluster2 = 7.0 + 0.5 * np.random.randn(remainder)
            locs = np.concatenate([cluster1, cluster2]).reshape(-1,1)
            np.random.shuffle(locs)
    else:
        if distribution_type == "Uniform":
            locs = np.random.uniform(0, 10, size=(num_arms, 2))
        else:  # Lumpy
            half = num_arms // 2
            remainder = num_arms - half
            cluster1_x = 3.0 + 0.5*np.random.randn(half)
            cluster1_y = 3.0 + 0.5*np.random.randn(half)
            cluster2_x = 7.0 + 0.5*np.random.randn(remainder)
            cluster2_y = 7.0 + 0.5*np.random.randn(remainder)
            locs = np.vstack([
                np.column_stack([cluster1_x, cluster1_y]),
                np.column_stack([cluster2_x, cluster2_y])
            ])
            np.random.shuffle(locs)
    return locs.astype(np.float32)

############################################################
# 2) HELPER: Subset selection policies for each round
############################################################
def pick_arms(method_choice, est_rewards, est_locations, subset_size):
    with torch.no_grad():
        rw = est_rewards.detach().cpu().numpy()
        loc = est_locations.detach().cpu().numpy()

    num_arms = len(rw)
    def coherence(subs):
        if len(subs)==0: 
            return 0
        chosen_loc = loc[subs]
        centroid = chosen_loc.mean(axis=0)
        dist = np.sqrt(np.sum((chosen_loc - centroid)**2, axis=1)).mean()
        return -dist  # smaller distance => bigger coherence

    def total_value(subs):
        return rw[subs].sum() + coherence(subs)

    if method_choice == "Greedy":
        chosen = []
        base_val = total_value(chosen)
        for _ in range(subset_size):
            best_gain, best_arm = -1e9, None
            for a in range(num_arms):
                if a in chosen:
                    continue
                tv = total_value(chosen + [a])
                gain = tv - base_val
                if gain>best_gain:
                    best_gain = gain
                    best_arm = a
            chosen.append(best_arm)
            base_val += best_gain
        return np.array(chosen)

    elif method_choice == "Greedy Swamp":
        chosen = pick_arms("Greedy", est_rewards, est_locations, subset_size).tolist()
        best_val = total_value(chosen)
        improved = True
        max_swaps=5
        nswaps=0
        while improved and nswaps<max_swaps:
            improved=False
            for arm_out in chosen:
                for arm_in in range(num_arms):
                    if arm_in in chosen:
                        continue
                    trial = chosen[:]
                    trial.remove(arm_out)
                    trial.append(arm_in)
                    val = total_value(trial)
                    if val>best_val:
                        chosen = trial
                        best_val=val
                        improved=True
                        break
                if improved:
                    break
            nswaps+=1
        return np.array(chosen)

    elif method_choice == "Sampled":
        best_sub, best_val = None, -1e9
        n_samples = min(50, max(1, int(comb(num_arms, subset_size, exact=False)//10)))
        for _ in range(n_samples):
            subs = np.random.choice(num_arms, subset_size, replace=False)
            val = total_value(subs)
            if val>best_val:
                best_val= val
                best_sub= subs
        return np.array(best_sub)

    elif method_choice == "Sampled Intel":
        round_idx = pick_arms.round_idx if hasattr(pick_arms, 'round_idx') else 0
        total_rounds = pick_arms.total_rounds if hasattr(pick_arms, 'total_rounds') else 200
        eps_start=1.0
        eps_min=0.05
        frac= round_idx/float(total_rounds)
        eps_t= eps_start*(1.0 - frac)
        eps_t= max(eps_t, eps_min)
        pick_arms.round_idx = round_idx+1
        if np.random.rand()< eps_t:
            return np.random.choice(num_arms, subset_size, replace=False)
        else:
            return pick_arms("Greedy", est_rewards, est_locations, subset_size)

    else:
        return np.random.choice(num_arms, subset_size, replace=False)

############################################################
# 3) The main Streamlit UI
############################################################
st.title("Coherence-based Combinatorial Multi-Armed Bandit (CMAB) Simulation")

# (A) Sidebar
st.sidebar.header("Simulation Setup")
num_arms = st.sidebar.slider("Number of Arms (N)", 4, 30, 5)
subset_size = st.sidebar.slider("Subset Size (K)", 1, num_arms, min(3,num_arms))
lambda_penalty = st.sidebar.number_input("Penalty parameter (λ)", 0.0, 10.0, 1.0)
space_dim = st.sidebar.selectbox("Latent Space Dimension", ["1D", "2D"])
state_space_known = st.sidebar.selectbox("State Space Knowledge", ["Known", "Unknown"])
agent_type = st.sidebar.selectbox("Agent Type", ["Generalist", "Specialist", "Both"])
distribution_type = st.sidebar.selectbox("Distribution Type", ["Uniform", "Lumpy"])
method_choice = st.sidebar.selectbox("Algorithm Method", ["Sampled", "Sampled Intel", "Greedy", "Greedy Swamp"])
num_rounds = st.sidebar.slider("Simulation Rounds", 10, 1000, 200)

# (B) User-defined rewards (mu)
st.header("Arm Parameters")
reward_of_arms = [
    st.number_input(f"Reward (μ) Arm {i+1}", 0.0, 5.0, 1.0, key=f"mu_{i}")
    for i in range(num_arms)
]

# (C) Real latent locations
latent_locations = []
if state_space_known == "Known":
    st.subheader("Known Latent Locations")
    for i in range(num_arms):
        if space_dim == "1D":
            loc = st.slider(f"Location Arm {i+1}", 0.0, 10.0, float(i % 10)+0.5, key=f"loc_{i}")
            latent_locations.append([loc])
        else:
            loc_x = st.slider(f"X Arm {i+1}", 0.0, 10.0, float(i % 10)+0.5, key=f"locx_{i}")
            loc_y = st.slider(f"Y Arm {i+1}", 0.0, 10.0, float((i // 10) % 10)+0.5, key=f"locy_{i}")
            latent_locations.append([loc_x, loc_y])
    real_locations = torch.tensor(latent_locations, dtype=torch.float32)
else:
    st.subheader("Latent Locations are unknown, auto-generated.")
    gen_locs = generate_real_locations(num_arms, distribution_type, space_dim)
    real_locations = torch.tensor(gen_locs, dtype=torch.float32)

# (Helper) for chunked average over 20-round windows
def chunked_mean(curve, chunk_size=20):
    out = []
    for start in range(0, len(curve), chunk_size):
        block = curve[start:start+chunk_size]
        out.append(np.mean(block))
    return out

if st.button("Run Simulation"):
    pick_arms.total_rounds = num_rounds
    pick_arms.round_idx = 0

    def create_agent():
        est_r = torch.rand(num_arms, requires_grad=True)
        est_l = torch.rand((num_arms, real_locations.shape[1]), requires_grad=True)
        est_lambda = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)
        est_noise = torch.randn(num_arms, requires_grad=True)
        return est_r, est_l, est_lambda, est_noise

    # Setup agents
    agent_data = []
    if agent_type == "Generalist":
        agent_data.append(("Generalist", create_agent()))
    elif agent_type == "Specialist":
        agent_data.append(("Specialist", create_agent()))
    else:
        agent_data.append(("Generalist", create_agent()))
        agent_data.append(("Specialist", create_agent()))

    agent_results = {}
    for (name, ag) in agent_data:
        agent_results[name] = {
            "rewards_history": [],
            "cumulative_rewards": [],
            "regret_history": [],
            "arms_selected": [],
            "est_rewards_hist": [],
            "est_lambda_hist": [],
            "est_locations_hist": []
        }

    true_rewards_t = torch.tensor(reward_of_arms, dtype=torch.float32)

    def agent_step(agent_tuple):
        (er, el, elam, ee) = agent_tuple
        chosen = pick_arms(method_choice, er, el, subset_size)
        with torch.no_grad():
            used_locations = real_locations if state_space_known=="Known" else el
            csel = used_locations[chosen]
            centroid = csel.mean(dim=0)
            dist = torch.norm(csel - centroid, dim=1).mean()
        # environment payoff
        r = true_rewards_t[chosen].sum() - (lambda_penalty * dist) + ee[chosen].sum()
        return chosen, r

    # single optimizer
    all_params = []
    for (_, (er,el,elam,ee)) in agent_data:
        all_params += [er, el, elam, ee]
    optimizer = optim.Adam(all_params, lr=0.01)

    # best possible ignoring distance
    topk_val = torch.topk(true_rewards_t, subset_size).values.sum().item()

    for t in range(num_rounds):
        # forward pass => store environment payoff for each agent
        agent_round_info = []
        for (name, (er,el,elam,ee)) in agent_data:
            chosen, env_r = agent_step((er,el,elam,ee))
            agent_round_info.append((name, chosen, env_r))

        # backprop => (predicted - env_r)**2
        total_loss = None
        for idx, (name, (er,el,elam,ee)) in enumerate(agent_data):
            chosen, env_r = agent_round_info[idx][1], agent_round_info[idx][2]
            used_locations = real_locations if state_space_known=="Known" else el
            csel = used_locations[chosen]
            centroid = csel.mean(dim=0)
            dist = torch.norm(csel - centroid, dim=1).mean()

            # agent's predicted payoff
            pred_r = er[chosen].sum() - (elam*dist) + ee[chosen].sum()
            loss_expr = (pred_r - env_r.detach())**2

            total_loss = loss_expr if total_loss is None else (total_loss + loss_expr)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # record logs
        for idx, (name, (er,el,elam,ee)) in enumerate(agent_data):
            chosen, env_r = agent_round_info[idx][1], agent_round_info[idx][2]
            arr = agent_results[name]
            arr["rewards_history"].append(env_r.item())
            arr["cumulative_rewards"].append(
                env_r.item() + (arr["cumulative_rewards"][-1] if arr["cumulative_rewards"] else 0)
            )
            reg = topk_val - env_r.item()
            arr["regret_history"].append(reg)
            arr["arms_selected"].append(chosen.tolist())
            arr["est_rewards_hist"].append(er.clone().detach().tolist())
            arr["est_lambda_hist"].append(elam.item())
            arr["est_locations_hist"].append(el.clone().detach().tolist())

    # Download CSV
    rows = []
    for (name, data_dict) in agent_results.items():
        for round_idx in range(num_rounds):
            row = {
                "Round": round_idx+1,
                "Agent": name,
                "SelectedArms": data_dict["arms_selected"][round_idx],
                "Reward": data_dict["rewards_history"][round_idx],
                "CumulativeReward": data_dict["cumulative_rewards"][round_idx],
                "Regret": data_dict["regret_history"][round_idx],
                "EstimatedLambda": data_dict["est_lambda_hist"][round_idx],
                "EstimatedRewards": str(data_dict["est_rewards_hist"][round_idx]),
                "EstimatedLocations": str(data_dict["est_locations_hist"][round_idx])
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df["RealLambda"] = lambda_penalty
    df["RealRewards"] = str(reward_of_arms)
    df["RealLocations"] = str(real_locations.tolist())

    st.download_button(
        label="Download Round-by-Round CSV",
        data=df.to_csv(index=False),
        file_name="simulation_rounds.csv",
        mime="text/csv"
    )

    # 1) Agent Comparison: Round-by-Round Rewards (with max possible)
    fig1, ax1 = plt.subplots()
    for (name, data_dict) in agent_results.items():
        ax1.plot(data_dict["rewards_history"], label=f"{name} Reward")
    # dotted line for max possible
    ax1.axhline(topk_val, linestyle='--', color='k', label='Max Possible')
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Reward")
    ax1.set_title("Agent Reward Comparison Over Time")
    ax1.legend()
    st.pyplot(fig1)

    # 2) Cumulative Rewards (existing)
    fig2, ax2 = plt.subplots()
    for (name, data_dict) in agent_results.items():
        ax2.plot(data_dict["cumulative_rewards"], label=f"{name} Cumulative")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Cumulative Reward")
    ax2.set_title("Cumulative Rewards")
    ax2.legend()
    st.pyplot(fig2)

    # 3) Cumulative Regret (existing)
    fig3, ax3 = plt.subplots()
    for (name, data_dict) in agent_results.items():
        ax3.plot(np.cumsum(data_dict["regret_history"]), label=f"{name} Cumulative Regret")
    ax3.set_xlabel("Round")
    ax3.set_ylabel("Cumulative Regret")
    ax3.set_title("Cumulative Regret Over Time")
    ax3.legend()
    st.pyplot(fig3)

    # 4) Round-by-Round Regret (new)
    figR, axR = plt.subplots()
    for (name, data_dict) in agent_results.items():
        axR.plot(data_dict["regret_history"], label=f"{name} Regret")
    axR.set_xlabel("Round")
    axR.set_ylabel("Regret")
    axR.set_title("Round-by-Round Regret")
    axR.legend()
    st.pyplot(figR)

    # 5) Windowed average reward (20-round) vs. max possible
    figW, axW = plt.subplots()
    for (name, data_dict) in agent_results.items():
        chunked_vals = chunked_mean(data_dict["rewards_history"], 20)
        axW.plot(chunked_vals, label=f"{name} Windowed(20) Avg")
    axW.axhline(topk_val, linestyle='--', color='k', label="Max possible")
    axW.set_xlabel("Window Index (each = 20 rounds)")
    axW.set_ylabel("Average Reward")
    axW.set_title("20-Round Windowed Average Reward")
    axW.legend()
    st.pyplot(figW)

    # 6) Per-agent Arm Reward Estimations Over Time
    for (name, data_dict) in agent_results.items():
        fig_est, ax_est = plt.subplots()
        rounds_len = len(data_dict["est_rewards_hist"])
        for arm_i in range(num_arms):
            est_i = [ data_dict["est_rewards_hist"][r][arm_i] for r in range(rounds_len) ]
            line = ax_est.plot(est_i, label=f"Arm{arm_i+1}")
            color = line[0].get_color()  # match color for real μ
            # dotted line for the true reward of this arm
            true_val = reward_of_arms[arm_i]
            ax_est.axhline(true_val, color=color, linestyle='--', alpha=0.7)

        ax_est.set_title(f"{name} - Arm Reward Estimations Over Time (dotted = true μ)")
        ax_est.set_xlabel("Round")
        ax_est.set_ylabel("Estimated Reward")
        ax_est.legend()
        st.pyplot(fig_est)

    # 7) Each agent's estimated lambda (with dotted line for real λ)
    for (name, data_dict) in agent_results.items():
        fig_lam, ax_lam = plt.subplots()
        ax_lam.plot(data_dict["est_lambda_hist"], label="Estimated λ")
        # dotted line for real λ
        ax_lam.axhline(lambda_penalty, color='red', linestyle='--', label="True λ")

        ax_lam.set_title(f"{name} - Lambda Estimate Over Time")
        ax_lam.set_xlabel("Round")
        ax_lam.set_ylabel("Lambda")
        ax_lam.legend()
        st.pyplot(fig_lam)

    st.success("Simulation completed successfully!")
