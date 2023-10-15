def g(x):
    return 5.58847218e-13 * x**2 - 3.08249938e-06 * x + 9.80663919


def sound_speed(x):
    return -2.43265270e-8 * x**2 - 3.82778332e-3 * x + 340.289478


def ro(x):
    return 4.14972281e-9 * x**2 - 1.17790055e-4 * x + 1.22534716


def Cy(x):
    return 0.0834 * x + 0.3579


def Cx(x):
    return 0.0008 * x**2 - 0.0033 * x + 0.06


def P_h(x):
    return -2.0454545454545678e-05 * x ** 2 - 0.193636 * x + 7014.545455


def P_v(x):
    return -3409 * x**2 + 2131.82 * x + 4770
