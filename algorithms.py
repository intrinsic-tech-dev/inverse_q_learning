import numpy as np

def get_boltzman_distribution(q):
    (nS, nA) = q.shape
    bd = []
    for s in range(nS):
        bd.append([])
        for a in range(nA):bd[-1].append(np.exp(q[s][a]))
        
    bd = np.array(bd)
    action_sum = np.sum(bd, axis=1).reshape(-1, 1)
    bd = bd/action_sum
    return np.array(bd)

###### Algorithm 1 ######

def iavi(feature_matrix, transition_probabilities, action_probabilities, trajectories, conf):

    nS = feature_matrix.shape[0]
    nA = action_probabilities.shape[1]

    gamma = conf['env']['gamma']

    epsilon_for_log = 1e-4
    # initialize tables for reward function and value function.
    r = np.zeros((nS, nA))
    q = np.zeros((nS, nA))

    r_diff_list = []

    # compute reverse topological order.
    T = []
    for i in reversed(range(nS)):
        T.append([i])

    # do while change in r over iterations is larger than theta.
    diff = np.inf
    while diff > conf['iavi']['theta']:
        print(diff)
        diff = 0
        for t in T[0:]:
            for i in t:
                # compute coefficient matrix X_A(s) as in Eq. (9).
                X = []
                for a in range(nA):
                    row = np.ones(nA)
                    for oa in range(nA):
                        if oa == a:
                            continue
                        row[oa] /= -(nA-1)
                    X.append(row)
                X = np.array(X)

                # compute target vector Y_A(s) as in Eq. (9).
                y = []
                for a in range(nA):
                    other_actions = [oa for oa in range(nA) if oa != a]
                    sum_of_oa_logs = np.sum([np.log(action_probabilities[i][oa] + epsilon_for_log) for oa in other_actions])
                    sum_of_oa_q = np.sum([transition_probabilities[i][oa] * gamma * np.max(q[np.arange(nS)], axis=1) for oa in other_actions])
                    y.append(np.log(action_probabilities[i][a] + epsilon_for_log)-(1/(nA-1))*sum_of_oa_logs+(1/(nA-1))*sum_of_oa_q-np.sum(transition_probabilities[i][a] * gamma * np.max(q[np.arange(nS)], axis=1)))
                y = np.array(y)

                # Find least-squares solution.
                x = np.linalg.lstsq(X, y, rcond=None)[0]
                    
                for a in range(nA):
                    diff = max(np.abs(r[i, a]-x[a]), diff)

                # compute new r and Q-values.
                r[i] = x
                for a in range(nA):
                    q[i, a] = r[i, a] + np.sum(transition_probabilities[i][a] * gamma * np.max(q[np.arange(nS)], axis=1))
    
    boltzman_distribution = get_boltzman_distribution(q)

    return q, r_diff_list, boltzman_distribution


###### Algorithm 2 ######

def iql(trajectories, conf):

    nS = conf['env']['grid_size']**2
    nA = conf['env']['num_actions']
    gamma = conf['env']['gamma']

    epochs = conf['iql']['epochs']
    alpha_r = conf['iql']['alpha_r']
    alpha_q = conf['iql']['alpha_q']
    alpha_sh = conf['iql']['alpha_sh']

    r = np.zeros((nS, nA))
    q = np.zeros((nS, nA))
    q_sh = np.zeros((nS, nA))
    state_action_visit_counter = np.zeros((nS, nA))

    epsilon_for_log = 1e-4
    r_diff_list = []
    diff = np.inf
    for i in range(epochs):

        if i%20 == 0:
            print(f"Epoch {i} | diff {diff}")
       
        for traj in trajectories:
            for (s, a, _, new_s) in traj:

                state_action_visit_counter[s][a] += 1
        
                q_sh[s, a] = (1-alpha_sh) * q_sh[s, a] + alpha_sh * (gamma * np.max(q[new_s])) # Eq 9
                state_action_visit_sum = np.sum(state_action_visit_counter[s])
                log_prob = np.log((state_action_visit_counter[s]/state_action_visit_sum) + epsilon_for_log)
                other_a = [0,1,2,3,4].remove(a)

                eta_a = log_prob[a] - q_sh[s][a] # For Equation 10
                eta_b = log_prob[other_a] - q_sh[s][other_a] # For Equation 6
                sum_oa = (1/(nA-1)) * np.sum(r[s][other_a] - eta_b) # For Equation 6
                
                r_ = (1-alpha_r) * r[s][a] + alpha_r * (eta_a + sum_oa)
                diff = np.abs(r[s][a]-r_)/ alpha_r
                r[s][a] = r_ 
                q[s, a] = (1-alpha_q) * q[s, a] + alpha_q * (r[s, a] + gamma  * np.max(q[new_s])) 
                s = new_s
        
    boltzman_distribution = get_boltzman_distribution(q)
    
    return q, r_diff_list, boltzman_distribution