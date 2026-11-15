# Third-party attribution

ToolFlow builds on [amazon-agi/tau2-bench-verified](https://github.com/amazon-agi/tau2-bench-verified), pinned to commit `864350a8971a8f8ee9e7b8472e2edc380a806b0c` in [upstream.lock.json](upstream.lock.json).

The upstream project provides the Retail domain, policy, simulated data, tools, message types, user simulator, original LLMAgent baseline, Orchestrator and Evaluator. These are not original ToolFlow implementations. The checkout is obtained separately; model weights and upstream data are not bundled in this repository.

ToolFlow's work is the model adapters, interactive runtime, evidence-based guards, confirmation and readback, bounded recovery, extended retail persistence, Web interface, benchmark integration and tests. References to upstream tasks in tests are offline regression cases, not runtime instructions or a measurement of LLM capability.

The upstream MIT notice follows. Preserve it when redistributing upstream code or substantial portions. Other installed Python packages and model weights remain subject to their respective licenses; this repository's MIT license does not relicense them.

```text
MIT License

Copyright (c) 2025 Sierra Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
