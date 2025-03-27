# 🎯 A/B Testing with Epsilon-Greedy & Thompson Sampling

This project implements two classic Multi-Armed Bandit algorithms — **Epsilon-Greedy** and **Thompson Sampling** — to simulate an A/B testing scenario with four advertisement options.

---

## 📌 Scenario

- **Bandits (Ads):** 4 choices with fixed rewards: `[1, 2, 3, 4]`
- **Trials:** 20,000 rounds per algorithm
- **Epsilon-Greedy:** Decaying epsilon with `ε = 1 / t`
- **Thompson Sampling:** Gaussian-based sampling with known precision

---

## 📊 Outputs

Each algorithm:
- Stores results in a CSV (`{Bandit, Reward, Algorithm}`)
- Prints **average reward** and **cumulative regret** via `loguru`
- Visualizes:
  - 📈 Learning process (average reward over time)
  - 📉 Comparison of cumulative rewards and regrets

---

## 📁 Files

| File | Description |
|------|-------------|
| `Bandit.py` | Main script with algorithm classes and plotting |
| `epsilon_greedy_rewards.csv` | Results from Epsilon-Greedy |
| `thompson_sampling_rewards.csv` | Results from Thompson Sampling |

---

## ▶️ How to Run

> Make sure you're in the virtual environment and required packages are installed:

```bash
pip install loguru matplotlib pandas
# AB-Testing
