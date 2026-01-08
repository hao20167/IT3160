# IT3160 - Introduction to Artificial Intelligence 
**Group 14**

| Student Name | Student ID |
| :--- | :--- |
| **Bùi Công Hào** | 202400042 |
| **Nguyễn Khắc Duy** | 202400100 |

> **A sophisticated Deep Reinforcement Learning framework designed for high-frequency trading.**

We employs a **hierarchical architecture** where a high-level "Router" agent dynamically selects specialized low-level "Trader" agents based on market volatility, trend analysis, and real-time risk metrics. This approach decouples macro-strategy from micro-execution, allowing for robust performance in the volatile BTC/USDT market.

---

## 🧠 Core Architecture

The system operates on two distinct temporal resolutions, leveraging a "Divide and Conquer" strategy:

### 1. High-Level Agent (The Strategist)
- **Role:** Market regime classification and model selection.
- **Input:** Macro-scale technical indicators (Minute-level features).
- **Action:** Selects the optimal low-level agent (policy) for the current market state.
- **Location:** `model/high_level/` & `model/pick_agent/`

### 2. Low-Level Agents (The Executors)
- **Role:** Precise trade execution and position management.
- **Input:** Micro-scale order book data and technical indicators (Second-level features).
- **Action:** Buy, Sell, or Hold actions with risk-adjusted sizing.
- **Algorithm:** Double DQN (DDQN) with **PES (Prioritized Experience Replay with Risk Awareness)**.
- **Location:** `model/low_level/`

---

## ✨ Key Features

- **Risk-Aware Experience Replay:** Prioritizes training samples not just by TD-error, but by associated risk metrics, ensuring the model learns to avoid catastrophic drawdowns.
- **Hierarchical Decision Making:** Switches strategies dynamically (e.g., from "Trend Following" to "Mean Reversion") based on high-level market signals.
- **Comprehensive Backtesting:** Built-in environment for validating strategies against historical data with transaction costs and slippage simulation.

---

## 🛠 Experimental Parameters

Below are the default hyperparameters used in the training process for both hierarchical levels.

| Parameter | Low-Level Agent | High-Level Agent | Description |
| :--- | :--- | :--- | :--- |
| `buffer_size` | 1,000,000 | 100,000 | Experience replay memory capacity |
| `batch_size` | 2,048 | 512 | Number of samples per training update |
| `hidden_nodes` | 128 | 128 | Neurons in the network hidden layers |
| `lr_init` | 1e-2 | 1e-2 | Initial learning rate for Adam optimizer |
| `gamma` | 1.0 | 1.0 | Discount factor for future rewards |
| `tau` | 0.005 | 0.005 | Soft target network update coefficient |
| `reward_scale` | 30.0 | 10.0 | Scaling factor applied to raw rewards |
| `action_dim` | 5 | 5 | Number of discrete actions available |
| `transcation_cost`| 0.00015 | 0.00015 | Simulated fee per trade execution |
| `beta` | 100.0 | - | Boltzmann temperature for PES |
| `risk_bond` | 0.1 | - | Risk threshold for prioritized sampling |

---

## 📂 Project Structure

```bash
IT3160/
├── env/                  # Custom OpenAI Gym Environments
│   ├── high_level_env.py # Environment for the strategy selector
│   └── low_level_env.py  # Tick-level trading environment
├── model/                # Agent Implementations
│   ├── high_level/       # Macro-strategy agents (DQN)
│   ├── low_level/        # Execution agents (DDQN, Risk-Aware)
│   └── pick_agent/       # Agent selection logic
├── util/                 # Core Utilities
│   ├── net.py            # PyTorch Neural Network definitions (CNN, LSTM, MLP)
│   ├── replay_buffer.py  # Advanced buffers (Multi-step, Prioritized)
│   └── graph.py          # Visualization and plotting tools
├── tool/                 # Helper scripts
└── data/                 # Feature engineering and datasets
```

## ⚠️ Disclaimer

*This software is for educational and research purposes only. Do not use this for live trading without extensive testing and understanding of the risks involved. Financial markets are unpredictable, and past performance is not indicative of future results.*
