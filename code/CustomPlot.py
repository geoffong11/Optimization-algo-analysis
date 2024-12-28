from GDMomentum import custom_fun_gd_momentum
from GDAdagrad import custom_fun_gd_adagrad
from GDAdam import custom_fun_gd_adam
from GDRMSProp import custom_fun_gd_rmsprop
from GDMiniBatch import custom_fun_gd_vanilla
from graph import plot_all_loss_graph

def custom_function_gd_comparison(loss_fun, grad_fun, x_0,
                                  decaying_lr, const_lr,
                                  momentum_constant=0.9, rmsprop_constant=0.9,
                                  adam_constant_1=0.9, adam_constant_2=0.9,
                                  max_t=1500, std=0):
    final_loss_adagrad, adagrad_loss_per_epoch = custom_fun_gd_adagrad(loss_fun, grad_fun, x_0, const_lr, max_t=max_t, std=std)
    final_loss_momentum, momentum_loss_per_epoch = custom_fun_gd_momentum(loss_fun, grad_fun, x_0, decaying_lr,
                                                                          momentum_constant=momentum_constant, max_t=max_t, std=std)
    final_loss_adam, adam_loss_per_epoch = custom_fun_gd_adam(loss_fun, grad_fun, x_0, const_lr, adam_constant_1=adam_constant_1,
                                                              adam_constant_2=adam_constant_2, max_t=max_t,  std=std)
    final_loss_rmsprop, rmsprop_loss_per_epoch = custom_fun_gd_rmsprop(loss_fun, grad_fun, x_0, const_lr,
                                                                       rmsprop_constant=rmsprop_constant, max_t=max_t, std=std)
    final_loss_vanilla, vanilla_loss_per_epoch = custom_fun_gd_vanilla(loss_fun, grad_fun, x_0, decaying_lr, max_t=max_t, std=std)
    loss_dict = {"rmsprop": rmsprop_loss_per_epoch, "adam": adam_loss_per_epoch,
                 "vanilla": vanilla_loss_per_epoch, "momentum": momentum_loss_per_epoch,
                 "adagrad": adagrad_loss_per_epoch}
    
    print(f"loss for vanilla: {final_loss_vanilla}")
    print(f"loss for momentum: {final_loss_momentum}")
    print(f"loss for adagrad: {final_loss_adagrad}")
    print(f"loss for rmsprop: {final_loss_rmsprop}")
    print(f"loss for adam: {final_loss_adam}")
    plot_all_loss_graph(loss_dict)
