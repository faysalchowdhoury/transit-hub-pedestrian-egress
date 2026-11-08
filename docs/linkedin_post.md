My evacuation simulation said a metro platform design passed the 4-minute rule. The hand calculation said it failed. The hand calculation was closer to right.

I built a pedestrian microsimulation for an island platform: a vectorised social-force model with spatial hashing, a Dijkstra distance field for navigation and an A* routing API. I compared it with an NFPA 130-style capacity hand calculation across loads and stair layouts (50 runs).

At 1,600 people with three 2.4 m stairs:
🔹 Simulation: platform clears in 2.4 min ✅
🔹 Hand calculation: 5.3 min ❌

The hand calc was 1.6-2.5× more conservative everywhere. But when I traced the gap, most of it came from my model: it ends when people reach the stair entry at walking-flow rates (~90 p/m/min), while the code method charges the slower stair-climb capacity (55.5 p/m/min). The stairs are the real bottleneck, and I hadn't modelled them yet.

Three things worth sharing:
1️⃣ Check flows against data before trusting a crowd model. My first version pushed 3.4 people/m/s through a door, twice what experiments show. Adding a speed-density relation (Weidmann) brought it to 1.4-1.7.
2️⃣ The hand calc can't see geometry. Three 1.8 m stairs and two 2.4 m stairs are identical on paper, but the simulation cleared the three-stair layout 9% faster (shorter walks).
3️⃣ Know what your model can't do. Mine doesn't reproduce "faster-is-slower" clogging, so it shouldn't be used for crush scenarios. That's in the README.

Caveats: no stairs, escalators or train-unloading surge yet; NFPA parameters are configurable defaults, not a compliance submittal.

Repo: [link]

Station designers: how do you reconcile microsimulation with NFPA 130 calcs when the AHJ asks?

#TransitDesign #NFPA130 #PedestrianModeling #CrowdSimulation #TransportationEngineering #Python #Simulation
