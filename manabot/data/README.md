### Design Summary

We are taking a **flexible, two-pronged** approach to encoding game observations in Manabot:

1. **Separate Tensors for Cards and Permanents**:  
   - We produce distinct arrays for the card data and for the permanent data.  
   - This allows neural networks to treat them differently—e.g., a specialized “card embedding” layer for spells vs. a separate “battlefield embedding” layer for permanents.  
   - Both are padded to fixed maximum sizes, like `[max_cards, card_dim]` and `[max_permanents, perm_dim]`.

2. **Universal “Objects” Array**:  
   - In addition, we build one “universal” or “wasteful” array that merges **all** objects—players, cards, permanents—into a single list (e.g. `[max_total_objects, object_dim]`).  
   - Actions reference this array for embedding. For example, if an action focuses on a particular card/permanent, we can look up its “universal row” to embed it directly in the action tensor.  
   - This universal array is also handy for attention-based methods that want to handle every entity in a single pass.

3. **Global Features**:  
   - We produce a single `[global_dim]` vector that packs in all top-level info (turn number, phase, step, game_over, etc.).  
   - All enumerations (like phase, step) are one-hot encoded; flags (like `game_over`) are simple booleans.

4. **Actions**:  
   - Each action is encoded as one-hot for its `ActionEnum` type plus a partial embedding of the focus object(s).  
   - By referencing the universal objects array, we can embed the relevant object(s) inside the action vector.  
   - We pad to `[max_actions, action_dim]`.

5. **One-Hot for Enums**:  
   - All enumerations (phases, steps, zones, action types) are turned into short one-hot vectors.  
   - This avoids numeric illusions in the network.  
   - Over time, if the number of enumerations grows too large, we may adopt small learned embeddings.

6. **Embedding Future**:  
   - Currently, we do “on-the-fly” numeric encoding, but we anticipate migrating to partial learned embeddings if we see expansions in card IDs or large enumerations.  
   - Our code is structured to let us drop in these embedding lookups later without overhauling the entire architecture.

### Next Steps to Build

1. **Refine the Object-Specific Dimensions**:  
   - Decide precisely what fields go into each card/permanent array.  
   - Confirm the exact dimension (card_dim, perm_dim, universal_dim) and finalize any leftover padding.

2. **Add Extended Dimension Validation**:  
   - Expand the dimension checks to ensure arrays are always the size we expect.  
   - Possibly log warnings when we exceed some threshold or drop objects due to `max_cards`/`max_permanents` overflow.

3. **Implement Full Action Embedding**:  
   - We currently embed the first focus object as a partial example.  
   - For complex actions referencing multiple objects, we may want more flexible logic (e.g., embedding both source and target if relevant).

4. **Hook Up the Single-Layer Unifier**:  
   - We have a convenience method for generating a linear layer that transforms cards/perms to a shared dimension.  
   - Next, confirm the model code uses it effectively—for instance, combining the card/permanent embeddings in a single pipeline.

5. **Integrate Testing**:  
   - Write unit tests that feed in synthetic Observations and confirm the shape, dtype, and content of each output array.  
   - Optionally test with real environment data to ensure correctness and performance.

With these final touches—full dimension definitions, thorough testing, and finishing the C++ side’s reindexing—the data representation system will be ready for immediate RL experiments (like PPO) and future expansions (embedding large sets or mixing in advanced attention).