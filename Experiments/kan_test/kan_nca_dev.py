import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import sys
    sys.path.append('/home/alex/PhD/Differentiable-Patterning/')
    from Common.model.fast_kan import FastRBFKAN,FastRBFKANLayer,plot_fast_rbf_kan_edges,plot_top_fast_rbf_kan_edges
    from NCA.model.NCA_fast_KAN_model import FastKaNCA
    import jax
    import jax.numpy as np
    import matplotlib.pyplot as plt
    import jax.random as jr
    from einops import rearrange


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Test FastRBFKAN function
    """)
    return


@app.cell
def _():
    kan = FastRBFKAN(
        layers_hidden=[4,4,4],
        final_zero_init=False,
        key=jr.PRNGKey(50),
        num_basis=30,
        base_activation="silu",
        base_init_scale=0.05)

    return (kan,)


@app.cell
def _(kan):
    # x = jr.uniform(jr.PRNGKey(69), (100), minval=-1.0, maxval=1.0)

    # plt.plot(x[:,0])
    # plt.show()

    fig = plot_top_fast_rbf_kan_edges(
        kan,
        layer_index=0,
        k=4,
        # input_indices=[0, 1],
        # output_indices=[0,1],
        xs=np.linspace(-4, 4, 200),
    )
    fig

    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Test FastKaNCA
    """)
    return


@app.function
def reshape_to_image(X):
    return rearrange(X,"C X Y -> X Y C")


@app.cell
def _():
    X = jr.uniform(key=jr.PRNGKey(1235),shape=(3,32,32))
    # print(model(X).shape)
    plt.imshow(reshape_to_image(X))
    plt.show()
    model = FastKaNCA(N_CHANNELS=3,KAN_AUX={"final_zero_init":False})
    Y = model(X)
    print(np.sum(X-Y))
    plt.imshow(reshape_to_image(Y))
    plt.show()
    return (model,)


@app.cell
def _(model):
    diff,static = model.partition()
    return (static,)


@app.cell
def _(static):
    print(static)
    return


if __name__ == "__main__":
    app.run()
