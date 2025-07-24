class Constants:
    """
    Claude helped me ideate interesting test cases for the minimal routing and
    non-minimal routing schedulers.
    """

    # Graph settings
    DEFAULT_GRAPH = default_adjacency = [
        # Group 0 (all-to-all): Mixed producers and consumers
        ("C0", "P1"),
        ("C0", "P2"),
        ("C0", "C3"),
        ("P1", "C0"),
        ("P1", "P2"),
        ("P1", "C3"),
        ("P2", "C0"),
        ("P2", "P1"),
        ("P2", "C3"),
        ("C3", "C0"),
        ("C3", "P1"),
        ("C3", "P2"),
        # Group 1 (all-to-all): All consumers
        ("C4", "C5"),
        ("C4", "C6"),
        ("C4", "C7"),
        ("C5", "C4"),
        ("C5", "C6"),
        ("C5", "C7"),
        ("C6", "C4"),
        ("C6", "C5"),
        ("C6", "C7"),
        ("C7", "C4"),
        ("C7", "C5"),
        ("C7", "C6"),
        # Group 2 (all-to-all): All producers
        ("P8", "P9"),
        ("P8", "P10"),
        ("P8", "P11"),
        ("P9", "P8"),
        ("P9", "P10"),
        ("P9", "P11"),
        ("P10", "P8"),
        ("P10", "P9"),
        ("P10", "P11"),
        ("P11", "P8"),
        ("P11", "P9"),
        ("P11", "P10"),
        # Group 3 (all-to-all): Mixed producers and consumers
        ("P12", "C13"),
        ("P12", "P14"),
        ("P12", "C15"),
        ("C13", "P12"),
        ("C13", "P14"),
        ("C13", "C15"),
        ("P14", "P12"),
        ("P14", "C13"),
        ("P14", "C15"),
        ("C15", "P12"),
        ("C15", "C13"),
        ("C15", "P14"),
        # Global links between groups (following Dragonfly pattern)
        # Each group connects to other groups via designated global links
        ("C0", "C4"),
        ("C4", "C0"),  # Group 0 <-> Group 1
        ("P1", "P8"),
        ("P8", "P1"),  # Group 0 <-> Group 2
        ("P2", "P12"),
        ("P12", "P2"),  # Group 0 <-> Group 3
        ("C5", "P9"),
        ("P9", "C5"),  # Group 1 <-> Group 2
        ("C6", "C13"),
        ("C13", "C6"),  # Group 1 <-> Group 3
        ("P10", "P14"),
        ("P14", "P10"),  # Group 2 <-> Group 3
    ]

    # Data transfer settings
    DEFAULT_DATA_TRANSFERS = [
        # Scenario 1: Intra-group transfers that compete for same edges
        ("P1", "C3"),  # Within Group 0: P1 -> C3 (direct edge available)
        ("P2", "C3"),  # Within Group 0: P2 -> C3 (same destination, will pipeline)
        # Scenario 2: Cross-group transfers sharing bottleneck global links
        ("P1", "C5"),  # Group 0 -> Group 1: must use P1->C0->C4->C5 (minimal path)
        ("P2", "C6"),  # G0 -> G1: must use P2->C0->C4->C6 (same global link C0->C4)
        # Scenario 3: Producer group to consumer group (high bandwidth demand)
        ("P8", "C4"),  # Group 2 -> Group 1: P8->P9->C5->C4 (minimal)
        ("P9", "C7"),  # Group 2 -> Group 1: same global link bottleneck
        ("P10", "C6"),  # Group 2 -> Group 1: same global link bottleneck
        # Scenario 4: Cross-group to mixed group
        ("P11", "C13"),  # Group 2 -> Group 3: P11->P10->P14->C13 (minimal)
        ("P8", "C15"),  # Group 2 -> Group 3: different destination, same global link
    ]
