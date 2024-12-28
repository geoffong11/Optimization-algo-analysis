from GDMiniBatch import gradient_descent_vanilla
from GDMomentum import gradient_descent_momentum
from GDAdagrad import gradient_descent_adagrad
from GDAdam import gradient_descent_adam
from GDRMSProp import gradient_descent_rmsprop
from graph import plot_all_loss_graph

def dataset_comparison_between_descent(loss_fun, grad_fun, X, y, theta, decaying_lr, const_lr,
                                  momentum_constant=0.9, rmsprop_constant=0.9,
                                  adam_constant_1=0.9, adam_constant_2=0.9,
                                  batch_size=16, max_t=4):

    theta_vanilla, final_loss_vanilla, loss_vanilla_per_iter = gradient_descent_vanilla(loss_fun, grad_fun, X, y, theta,
                                            decaying_lr, max_t=max_t, batch_size=batch_size)
    theta_momentum, final_loss_momentum, loss_momentum_per_iter = gradient_descent_momentum(loss_fun, grad_fun, X, y, theta,
                                            momentum_constant, decaying_lr, max_t=max_t, batch_size=batch_size)
    theta_adagrad, final_loss_adagrad, loss_adagrad_per_iter = gradient_descent_adagrad(loss_fun, grad_fun, X, y, theta,
                                            const_lr, max_t=max_t, batch_size=batch_size)
    theta_rmsprop, final_loss_rmsprop, loss_rmsprop_per_iter = gradient_descent_rmsprop(loss_fun, grad_fun, X, y, theta, rmsprop_constant,
                                                const_lr, max_t=max_t, batch_size=batch_size)
    theta_adam, final_loss_adam, loss_adam_per_iter = gradient_descent_adam(loss_fun, grad_fun, X, y,
                                                                            theta, adam_constant_1, adam_constant_2,
                                                                            const_lr, max_t=max_t, batch_size=batch_size)
    loss_dict = {"rmsprop": loss_rmsprop_per_iter, "adam": loss_adam_per_iter, 
                 "vanilla": loss_vanilla_per_iter, "momentum": loss_momentum_per_iter,
                 "adagrad": loss_adagrad_per_iter}
    print(f"loss for vanilla: {final_loss_vanilla} theta: {theta_vanilla}")
    print(f"loss for momentum: {final_loss_momentum} theta: {theta_momentum}")
    print(f"loss for adagrad: {final_loss_adagrad} theta: {theta_adagrad}")
    print(f"loss for rmsprop: {final_loss_rmsprop} theta: {theta_rmsprop}")
    print(f"loss for adam: {final_loss_adam} theta: {theta_adam}")

    plot_all_loss_graph(loss_dict)
