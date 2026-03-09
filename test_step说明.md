# test 中 `tot_steps` 与 env.step 执行步数说明

## 结论（当前代码）
在 `ExperimentManager.test()` 里，循环条件是：

```python
while not done and step < tot_steps:
```

并且每次循环末尾都有：

```python
step += 1
```

因此：
- 当 `tot_steps=1` 时，`env.step(...)` **恰好执行 1 次**。
- 若环境自身提前终止（`done=True`），则会少于 `tot_steps`。

## 与环境终止条件的关系
环境在 `CompressorEnv.step()` 中的终止逻辑为：

```python
done = self.decision_count >= self.env_config.max_decisions
```

也就是说，单个 episode 的理论上限是 `max_decisions` 次决策步。

所以测试函数真正执行的步数是：

```text
实际步数 = min(tot_steps, 直到 done 触发前可执行的步数)
```

## 举例
- 若 `max_decisions=80` 且从 reset 后开始测试：
  - `tot_steps=1` -> 实际执行 1 步。
  - `tot_steps=10` -> 实际执行 10 步。
  - `tot_steps=100` -> 最多执行到 episode 结束（约 80 步）。
