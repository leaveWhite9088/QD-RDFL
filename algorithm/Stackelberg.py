import numpy as np
from scipy.optimize import root, minimize_scalar


class Stackelberg:

    # 领导者的效用函数
    @staticmethod
    def _leader_utility(eta, alpha, fn, x):
        # print(f"eta: {eta}, alpha: {alpha}, fn: {fn}, x: {x}")
        # print(f"fn type: {type(fn)}, x type: {type(x)}")
        if fn is None or x is None:
            raise ValueError("fn or x is None")

        S = np.dot(x, fn)
        return 0.5 * alpha * np.log(S + 1) - eta

    # 追随者的效用函数
    @staticmethod
    def _follower_utility(xn, eta, fn_n, S, lambda_, rho):
        return (xn * fn_n * eta) / S - lambda_ * rho * xn

    # 追随者的最佳反应函数
    @staticmethod
    def _followers_best_response(x, eta, fn, lambda_, rho):
        S = np.dot(x, fn)
        residuals = (fn * eta) / S - (x * fn ** 2 * eta) / S ** 2 - lambda_ * rho
        return residuals

    # 给定 eta，求解所有追随者的最佳反应 x
    @staticmethod
    def _solve_followers(eta, fn, lambda_, rho):
        N = len(fn)
        x0 = np.ones(N)  # 初始猜测/'/'"

        # 使用 root 函数求解非线性方程组
        sol = root(Stackelberg._followers_best_response, x0, args=(eta, fn, lambda_, rho), method='hybr')

        if sol.success:
            return sol.x
        else:
            return None

    # 领导者的优化目标函数（负效用，因为我们使用的是最小化函数）
    @staticmethod
    def _leader_objective(eta, alpha, fn, lambda_, rho):
        try:
            x = Stackelberg._solve_followers(eta, fn, lambda_, rho)
            U = Stackelberg._leader_utility(eta, alpha, fn, x)
            return -U
        except ValueError:
            return np.inf  # 如果无法求解追随者反应，返回无穷大

    # 寻找Stackelberg均衡
    @staticmethod
    def find_stackelberg_equilibrium(alpha, fn, lambda_, rho):
        # 定义优化范围，根据领导者效用函数，eta >=0
        bounds = (0, 1000)  # 根据具体情况调整上界

        # 使用 minimize_scalar 进行一维优化
        res = minimize_scalar(
            Stackelberg._leader_objective,
            bounds=bounds,
            args=(alpha, fn, lambda_, rho),
            method='bounded'
        )

        if res.success:
            eta_opt = res.x
            x_opt = Stackelberg._solve_followers(eta_opt, fn, lambda_, rho)
            U_opt = Stackelberg._leader_utility(eta_opt, alpha, fn, x_opt)
            return eta_opt, x_opt, U_opt
        else:
            raise ValueError("领导者的优化失败。")
