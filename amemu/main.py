"""
Author:
"""
from amemu.trainingpoints import pk_linear
from amemu.src.gp.training import train_gps


def main(nlhs: list):
    """
    We generate the training points using CLASS and we train the GPs.

    Args:
        nlhs (list): a list of the number of LH points used. These LH
        points are generated using the sampleLHS.R script.
    """
    for i in nlhs:
        print(f"Generating training points for {i} LH samples")
        cosmos, pkl = pk_linear("lhs_" + str(i))

        print(f"Training GPs for {i} LH samples")
        gps = train_gps(i, jitter=1e-10)

        # these are not used further
        del cosmos, pkl, gps


if __name__ == "__main__":
    main([500])
