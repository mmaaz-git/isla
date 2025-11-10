import isla as ia

def test_props():
    A = ia.array([
        [[1, 2], [-3, -1]],
        [[0, 1], [2, 4]]
    ])

    print(ia.props.is_Z_matrix(A))
    print(ia.props.is_M_matrix(A))
    print(ia.props.is_H_matrix(A))
    print(ia.props.is_strictly_diagonally_dominant(A))
    print(ia.props.is_strongly_regular(A))