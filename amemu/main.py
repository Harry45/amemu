"""
Author:
"""
from amemu.trainingpoints import pk_linear


def main(nlhs: list):
    """
    main _summary_

    Args:
        nlhs (list): _description_
    """
    for i in nlhs:
        print(f"Generating training points for {i} LH samples")
        cosmos, pkl = pk_linear("lhs_" + str(i))


if __name__ == "__main__":
    main([500])
