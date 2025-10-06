# (Graaduate level) Selected Topics in Reinforcement Learning – Fall 2023 (NYCU) - CSIC30163

This repository contains the coursework for the **Selected Topics in Reinforcement Learning** course offered at **National Yang Ming Chiao Tung University (NYCU)**, Fall 2023. The course covers fundamental and advanced concepts in reinforcement learning (RL), with hands-on lab projects focusing on classic games like **2048** and **Atari environments** using modern RL algorithms such as **TD(0)**, **DQN**, **Double DQN**, **Dueling DQN**, and **PPO**.

> Name: 林伯偉 (Bo-Wei Lin)  
> Student ID: 109612019  
> Labs: 3 Projects + Bonus Explorations  
> Instructor: Prof. I-Chen Wu 

---

## Lab 1 – TD(0) & n-Tuple Network for 2048

- **Goal:** Implement a reinforcement learning agent to solve the game **2048** using **n-tuple networks** and **TD(0)** learning.
- **Key Concepts:**
  - TD(0) bootstrapping with step-size α.
  - Breaking large state spaces using **n-tuple network**.
  - Action selection via averaging expected rewards across tuple evaluations.
- **Highlights:**
  - Learned from after-states with reverse-time updates.
  - Report includes TD backup diagrams and training methodology.
- 📄 [Lab1 Report PDF](./Lab1_report_109612019.pdf)
- 📄 [Lab1 Assignment Guide](./Lab1-Guide.pdf)

---

## Lab 2 – Deep Q-Learning for Atari (MsPacman, Enduro)

- **Goal:** Train a DQN-based agent to play Atari games.
- **Environments:**
  - `MsPacman-v5`
  - `Enduro-v5`
- **Explored Variants:**
  - **Double DQN** – reduced Q-value overestimation using separate online & target networks.
  - **Dueling DQN** – separated state-value and advantage for more robust learning.
  - **Parallelized Rollouts** – improved exploration via multiple simultaneous environments.
- **Visualization:** TensorBoard logs included for all model variants.
- 📄 [Lab2 Report PDF](./Lab2_report.pdf)
- 📄 [Lab2 Assignment Guide](./Lab2-Guide.pdf)

---

## Lab 3 – Proximal Policy Optimization (PPO) on Enduro-v5

- **Goal:** Implement PPO with GAE-Lambda for continuous improvement in **Enduro-v5**.
- **Key Takeaways:**
  - PPO is **on-policy**, updates bounded by clipped surrogate objective.
  - **GAE-λ** balances bias-variance in advantage estimation.
  - Parameter **λ** tunes the impact of long-term rewards.
- **Topics Discussed:**
  - Update stability with clipped ratio.
  - Lambda parameter effects on training and performance.
- 📄 [Lab3 Report PDF](./Lab3_report.pdf)
- 📄 [Lab3 Assignment Guide](./Lab3-Guide.pdf)

---

## 🛠️ Tools & Libraries

- Python 3.8+
- PyTorch 2.0.1
- OpenAI Gym (`gym==0.26.2`)
- TensorBoard (`tensorboard==2.14.0`)
- MoviePy, OpenCV for video output and visualization

---

## 📚 References

- Mnih et al., “Playing Atari with Deep Reinforcement Learning,” 2013.
- Schulman et al., “Proximal Policy Optimization Algorithms,” 2017.
- Official OpenAI Gym Docs: [https://gym.openai.com/docs/](https://gym.openai.com/docs/)
- PPO Tips: [https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
