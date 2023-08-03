from keras.layers import (  # type: ignore[import]
    BatchNormalization,
    Dense,
    Input,
    Conv1D,
    Add,
    ELU,
    Flatten,
    MaxPooling1D,
)

# from keras.layers import (
#     GlobalAveragePooling1D,
#     Softmax,
#     Concatenate,
#     Reshape,
#     Multiply,
#     ReLU,
# )
from keras.initializers import HeNormal  # type: ignore[import]

bias_initializer = HeNormal(seed=0)


def diluted_conv_block(inputs, feature_dim, batch_normalisation=False):
    # with K.name_scope(block_name)
    l1_p1 = Conv1D(
        filters=feature_dim,
        kernel_size=3,
        padding="same",
        dilation_rate=1,
        use_bias=True,
        bias_initializer=bias_initializer,
    )(inputs)
    l1_p2 = Conv1D(
        filters=feature_dim,
        kernel_size=3,
        padding="same",
        dilation_rate=1,
        use_bias=True,
        bias_initializer=bias_initializer,
    )(inputs)
    l1_add = Add()([l1_p1, l1_p2])
    l1_ELU = ELU()(l1_add)
    # second layer of the DCB
    # l2_p1 = Conv1D(filters=feature_dim, kernel_size=5, padding="same", dilation_rate=2, use_bias=True, bias_initializer=bias_initializer)(l1_ELU)
    # l2_p2 = Conv1D(filters=feature_dim, kernel_size=5, padding="same", dilation_rate=2, use_bias=True, bias_initializer=bias_initializer)(l1_ELU)
    l2_p1 = Conv1D(
        filters=feature_dim,
        kernel_size=3,
        padding="same",
        dilation_rate=2,
        use_bias=True,
        bias_initializer=bias_initializer,
    )(l1_ELU)
    l2_p2 = Conv1D(
        filters=feature_dim,
        kernel_size=3,
        padding="same",
        dilation_rate=2,
        use_bias=True,
        bias_initializer=bias_initializer,
    )(l1_ELU)
    l2_add = Add()([l2_p1, l2_p2])
    l2_ELU = ELU()(l2_add)
    # third layer of the DCB
    # l3_p1 = Conv1D(filters=feature_dim, kernel_size=9, padding="same", dilation_rate=4, use_bias=True, bias_initializer=bias_initializer)(l2_ELU)
    # l3_p2 = Conv1D(filters=feature_dim, kernel_size=9, padding="same", dilation_rate=4, use_bias=True, bias_initializer=bias_initializer)(l2_ELU)
    l3_p1 = Conv1D(
        filters=feature_dim,
        kernel_size=3,
        padding="same",
        dilation_rate=4,
        use_bias=True,
        bias_initializer=bias_initializer,
    )(l2_ELU)
    l3_p2 = Conv1D(
        filters=feature_dim,
        kernel_size=3,
        padding="same",
        dilation_rate=4,
        use_bias=True,
        bias_initializer=bias_initializer,
    )(l2_ELU)
    l3_add = Add()([l3_p1, l3_p2])
    l3_ELU = ELU()(l3_add)

    residual = Conv1D(filters=feature_dim, kernel_size=1, padding="same")(inputs)
    res_add = Add()([l3_ELU, residual])
    if batch_normalisation:
        bn = BatchNormalization()(res_add)
        return bn
    else:
        return res_add


def time_diluted_conv_net(feature_dim, input_layer, pool_size, pool_stride):
    dcb_1 = diluted_conv_block(input_layer, feature_dim[0], batch_normalisation=True)
    mp_1 = MaxPooling1D(pool_size=pool_size, strides=pool_stride, padding="valid")(
        dcb_1
    )
    dcb_2 = diluted_conv_block(mp_1, feature_dim[1], batch_normalisation=True)
    mp_2 = MaxPooling1D(pool_size=pool_size, strides=pool_stride, padding="valid")(
        dcb_2
    )
    dcb_3 = diluted_conv_block(mp_2, feature_dim[2], batch_normalisation=True)
    mp_3 = MaxPooling1D(pool_size=pool_size, strides=pool_stride, padding="valid")(
        dcb_3
    )
    dcb_4 = diluted_conv_block(mp_3, feature_dim[3], batch_normalisation=True)
    mp_4 = MaxPooling1D(pool_size=pool_size, strides=pool_stride, padding="valid")(
        dcb_4
    )
    dcb_5 = diluted_conv_block(mp_4, feature_dim[4], batch_normalisation=False)
    return dcb_5
    # return dcb_2
