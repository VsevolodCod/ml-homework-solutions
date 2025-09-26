import numpy as np

def generate_session(env, policy, t_max=10000):
    """
    Играет игру до конца или в течение t_max шагов.
    :param policy: массив формы [n_states,n_actions] с вероятностями действий
    :returns: список состояний, список действий и сумма наград
    """
    states, actions = [], []
    total_reward = 0.

    s, info = env.reset()

    for t in range(t_max):
        # Выбираем действие согласно политике
        a = np.random.choice(np.arange(policy.shape[1]), p=policy[s])
        
        # Выполняем действие в среде
        new_s, r, done, truncated, info = env.step(a)
        
        # Записываем состояние, действие и добавляем награду
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done or truncated:
            break
            
    return states, actions, total_reward

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Выбирает состояния и действия из игр с наградами >= процентиля
    :param states_batch: список списков состояний, states_batch[session_i][t]
    :param actions_batch: список списков действий, actions_batch[session_i][t]
    :param rewards_batch: список наград, rewards_batch[session_i]

    :returns: elite_states,elite_actions, оба одномерных списка состояний и соответствующих действий из элитных сессий
    """
    # Вычисляем пороговую награду
    threshold = np.percentile(rewards_batch, percentile)
    
    elite_states = []
    elite_actions = []
    
    # Проходим по всем сессиям
    for i in range(len(rewards_batch)):
        if rewards_batch[i] >= threshold:
            # Добавляем все состояния и действия из этой элитной сессии
            elite_states.extend(states_batch[i])
            elite_actions.extend(actions_batch[i])
            
    return elite_states, elite_actions

def update_policy(elite_states, elite_actions, n_states, n_actions):
    """
    По старой политике и списку элитных состояний/действий из select_elites,
    возвращает новую обновленную политику, где каждая вероятность действия пропорциональна

    policy[s_i,a_i] ~ #[количество вхождений si и ai в элитных состояниях/действиях]

    :param elite_states: одномерный список состояний из элитных сессий
    :param elite_actions: одномерный список действий из элитных сессий
    """
    new_policy = np.zeros((n_states, n_actions))
    
    # Подсчитываем количество вхождений каждой пары состояние-действие
    for state, action in zip(elite_states, elite_actions):
        new_policy[state, action] += 1
    
    # Нормализуем для получения вероятностей
    for state in range(n_states):
        if np.sum(new_policy[state]) > 0:
            new_policy[state] /= np.sum(new_policy[state])
        else:
            # Если состояние никогда не посещалось, используем равномерное распределение
            new_policy[state] = np.ones(n_actions) / n_actions
            
    return new_policy