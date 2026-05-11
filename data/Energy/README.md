## Energy Scheduling Data

The energy benchmark expects the scheduling instances from the
**PredOpt Benchmarks** repository to be placed under `SchedulingInstances/`:

  https://github.com/PredOpt/predopt-benchmarks/tree/main/Energy

Expected local files:

- `SchedulingInstances/prices2013.dat`: 2013 day-ahead electricity price series.
- `SchedulingInstances/load{1,2,3}/day01.txt`: representative scheduling
  instance for each of three load profiles. The energy benchmark in this
  repository uses the `load3` profile.

These files are required to run `python scripts/paper_energy.py` and the
`configs/energy.yaml` benchmark; the synthetic `topk`, `shortest_path`, and
`pricing` benchmarks do not need this data.

### Upstream license

PredOpt Benchmarks is released under the MIT License. The upstream copyright
notice is reproduced below as required by that license.

```
MIT License

Copyright (c) 2021 PredOpt-Benchmarks

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
