from dataset.synth.tract import Tractogram, ParcellatedCylinder, Bundle, ControlPoint


def create_fibercup(radius=64, depth=5, margin=0.3, mult=500):

    num_nodes = 12
    cp_var = 5

    parcels = ParcellatedCylinder(num_nodes, radius, depth, margin=margin)

    tractogram = Tractogram()

    # ##########################

    edge = (3, 2)
    weight = 20
    cps = [
        ControlPoint((-int(0.32 * radius), int(0.6 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((-int(0.25 * radius), int(0.4 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.0 * radius), int(0.3 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.25 * radius), int(0.4 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.32 * radius), int(0.6 * radius), int(0.5 * depth)), cp_var)
    ]
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    bundle = Bundle(node0, node1, cps, num_streams)
    tractogram.add(edge, bundle)

    # ##########

    edge = (4, 9)
    weight = 30
    cp = [
        ControlPoint((-int(0.5 * radius), -int(0. * radius), int(0.5 * depth)), cp_var),
        ControlPoint((-int(0.25 * radius), -int(0.6 * radius), int(0.5 * depth)), cp_var)
    ]

    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    tractogram.add(edge, Bundle(node0, node1, cp, num_streams))

    # ##########

    edge = (6, 7)
    weight = 20
    cps = [
        ControlPoint((-int(0.65 * radius), -int(0.3 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((-int(0.5 * radius), -int(0.4 * radius), int(0.5 * depth)), cp_var),
        # ControlPoint((-int(0.48 * radius), -int(0.45 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((-int(0.5 * radius), -int(0.5 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((-int(0.6 * radius), -int(0.7 * radius), int(0.5 * depth)), cp_var)
    ]
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    tractogram.add(edge, Bundle(node0, node1, cps, num_streams))

    # ##########

    edge = (5, 0)
    weight = 20
    cps = []
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    tractogram.add(edge, Bundle(node0, node1, cps, num_streams))

    # ##########

    edge = (8, 1)
    weight = 20
    cps = [
        ControlPoint((int(0. * radius), -int(0.6 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.2 * radius), -int(0.4 * radius), int(0.5 * depth)), cp_var)
    ]
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    tractogram.add(edge, Bundle(node0, node1, cps, num_streams))

    # ##########

    edge = (8, 0)
    weight = 20
    cps = [
        ControlPoint((int(0. * radius), -int(0.6 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.2 * radius), -int(0.4 * radius), int(0.5 * depth)), cp_var)
    ]
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    tractogram.add(edge, Bundle(node0, node1, cps, num_streams))

    # ##########

    edge = (8, 11)
    weight = 20
    cps = [
        ControlPoint((int(0. * radius), -int(0.6 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.2 * radius), -int(0.4 * radius), int(0.5 * depth)), cp_var)

    ]
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    tractogram.add(edge, Bundle(node0, node1, cps, num_streams))

    return tractogram, parcels


def create_fibercup_classification(radius=64, depth=5, margin=0.3, mult=500):

    num_nodes = 12
    cp_var = 5

    parcels = ParcellatedCylinder(num_nodes, radius, depth, margin=margin)

    tractogram = Tractogram()

    # ##########################

    edge = (3, 2)
    weight = 20
    cps = [
        ControlPoint((-int(0.32 * radius), int(0.6 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((-int(0.25 * radius), int(0.4 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.0 * radius), int(0.3 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.25 * radius), int(0.4 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.32 * radius), int(0.6 * radius), int(0.5 * depth)), cp_var)
    ]
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    bundle = Bundle(node0, node1, cps, num_streams)
    tractogram.add(edge, bundle)

    # ##########

    edge = (4, 9)
    weight = 30
    cp = [
        ControlPoint((-int(0.5 * radius), -int(0. * radius), int(0.5 * depth)), cp_var),
        ControlPoint((-int(0.25 * radius), -int(0.6 * radius), int(0.5 * depth)), cp_var)
    ]

    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    tractogram.add(edge, Bundle(node0, node1, cp, num_streams))

    # ##########

    edge = (6, 7)
    weight = 20
    cps = [
        ControlPoint((-int(0.65 * radius), -int(0.3 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((-int(0.5 * radius), -int(0.4 * radius), int(0.5 * depth)), cp_var),
        # ControlPoint((-int(0.48 * radius), -int(0.45 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((-int(0.5 * radius), -int(0.5 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((-int(0.6 * radius), -int(0.7 * radius), int(0.5 * depth)), cp_var)
    ]
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    tractogram.add(edge, Bundle(node0, node1, cps, num_streams))

    # ##########

    edge = (6, 1)
    weight = 20
    cps = []
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    tractogram.add(edge, Bundle(node0, node1, cps, num_streams))

    # ##########

    edge = (8, 1)
    weight = 20
    cps = [
        ControlPoint((int(0. * radius), -int(0.6 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.2 * radius), -int(0.4 * radius), int(0.5 * depth)), cp_var)
    ]
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    tractogram.add(edge, Bundle(node0, node1, cps, num_streams))

    # ##########

    edge = (8, 0)
    weight = 20
    cps = [
        ControlPoint((int(0. * radius), -int(0.6 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.2 * radius), -int(0.4 * radius), int(0.5 * depth)), cp_var)
    ]
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    tractogram.add(edge, Bundle(node0, node1, cps, num_streams))

    # ##########

    edge = (8, 11)
    weight = 20
    cps = [
        ControlPoint((int(0. * radius), -int(0.6 * radius), int(0.5 * depth)), cp_var),
        ControlPoint((int(0.2 * radius), -int(0.4 * radius), int(0.5 * depth)), cp_var)

    ]
    node0 = parcels.nodes[edge[0]]
    node1 = parcels.nodes[edge[1]]
    num_streams = weight * mult

    tractogram.add(edge, Bundle(node0, node1, cps, num_streams))

    return tractogram, parcels