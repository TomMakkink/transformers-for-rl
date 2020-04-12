import deepmind_lab 

# TODO: Consider creating a local cache to speed things up 
# TODO: Write a wrapper for the environment, so they behave like a openai gym environment 

DEFAULT_ACTION_SET = [
    np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.intc),    # Forward
    np.array([0, 0, 0, -1, 0, 0, 0], dtype=np.intc),   # Backward
    np.array([0, 0, -1, 0, 0, 0, 0], dtype=np.intc),   # Strafe Left
    np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.intc),    # Strafe Right
    np.array([-20, 0, 0, 0, 0, 0, 0], dtype=np.intc),  # Look Left
    np.array([20, 0, 0, 0, 0, 0, 0], dtype=np.intc),   # Look Right
    np.array([-20, 0, 0, 1, 0, 0, 0], dtype=np.intc),  # Look Left + Forward
    np.array([20, 0, 0, 1, 0, 0, 0], dtype=np.intc),   # Look Right + Forward
    np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.intc),    # Fire.
]