# Trust and SRSR review checklist

This version is based on `JNCA_repro_v2_pre_srsr`.  Its `base/controller.py`
and PPO action-mask path are intentionally unchanged.

## Invariants

1. Candidate-node filtering remains `generate_action_mask -> find_candidate_nodes
   -> check_node_constraints`.  The preserved controller checks all node
   attributes, therefore static node security satisfies `Sl(node) >= Sdl(VNF)`.
   Link path checking likewise retains bandwidth and static link-security checks.
2. `security` attributes are never consumed: `Controller.update_resource`
   returns immediately for that attribute.  Place, release, route, and undo
   therefore leave every physical security level unchanged.
3. Trust is in `[0, 1]`:
   `0.6 * Sl/6 + 0.4 * (Suc + 1)/(Cou + 2)`.  The Laplace term prevents a
   zero-denominator and assigns an untried node neutral feedback of `0.5`.
4. Each completed SFC updates each distinct selected physical node once.  A
   success increments both `Cou` and `Suc`; a failed partial mapping increments
   only `Cou`.  Early rejection selects no node and changes no trust counter.
5. SRSR is snapshotted before the current request updates trust.  The rate is
   calculated over deployed security-sensitive requests, while acceptance rate
   remains the separate admission-capacity metric.

## Required pre-training checks

```bash
python verify_security_accounting.py
python -m py_compile config.py base/counter.py base/environment.py base/scenario.py data/physical_network.py
```

Then run one short CPU smoke test (five requests).  In its record CSV verify:

* all accepted mappings satisfy static node/link security;
* `security_sensitive_request_count` is no greater than `success_count`;
* `srsr` is in `[0, 1]` and is not mechanically forced to one; and
* physical node/link `security` values match their initial values after all
  requests have left.

Only after those checks pass should five-seed GPU experiments begin.
