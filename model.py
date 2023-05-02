import numpy
from scipy import optimize
from astropy.io import fits


def make_2d_gaussian_1d_repr(xy, x0, y0, x_compr, y_compr, max_intensity, position_angle):
    sigma_x = 1 / x_compr
    sigma_y = 1 / y_compr
    f = max_intensity * 2 * sigma_x * sigma_y
    (x, y) = xy
    x0 = float(x0)
    y0 = float(y0)
    a = (numpy.cos(position_angle) ** 2) / (2 * sigma_x ** 2) + (numpy.sin(position_angle) ** 2) / (2 * sigma_y ** 2)
    b = -(numpy.sin(2 * position_angle)) / (4 * sigma_x ** 2) + (numpy.sin(2 * position_angle)) / (4 * sigma_y ** 2)
    c = (numpy.sin(position_angle) ** 2) / (2 * sigma_x ** 2) + (numpy.cos(position_angle) ** 2) / (2 * sigma_y ** 2)
    intensity = f / 2 / sigma_x / sigma_y * numpy.exp(-(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)))
    return intensity.ravel()


def flux(compr, max_intensity):
    sigma_a = 1 / compr[0]
    sigma_b = 1 / compr[1]
    f = max_intensity * 2 * sigma_a * sigma_b
    return f


def make_model(image, model, init_parameters):
    repr_img = numpy.ravel(image)
    x = numpy.linspace(0, image.shape[0], image.shape[0])
    y = numpy.linspace(0, image.shape[1], image.shape[1])
    x, y = numpy.meshgrid(x, y)
    popt, pcov = optimize.curve_fit(model, (x, y), repr_img, init_parameters)
    data_fitted = model((x, y), *popt)
    data_fitted = numpy.reshape(data_fitted, newshape=image.shape)
    return popt, data_fitted


def five_gaussian_model(xy, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30):
    res = make_2d_gaussian_1d_repr(xy, p1, p2, p3, p4, p5, p6) + \
        make_2d_gaussian_1d_repr(xy, p7, p8, p9, p10, p11, p12) + \
        make_2d_gaussian_1d_repr(xy, p13, p14, p15, p16, p17, p18) + \
        make_2d_gaussian_1d_repr(xy, p19, p20, p21, p22, p23, p24) + \
        make_2d_gaussian_1d_repr(xy, p25, p26, p27, p28, p29, p30)
    return res


img = fits.open("data.fits")[0].data
res_params, res_img = make_model(img, five_gaussian_model, [100, 100, 1, 1, 1, 0] * 5)
test_fits = fits.PrimaryHDU(data=res_img - img)
fits.HDUList([test_fits]).writeto("residual.fits", overwrite=True)
for i in range(5):
    print(f"{i + 1}) center = ({res_params[i * 6]}, {res_params[i * 6 + 1]})")
    print(f"sigma = ({1 / res_params[i * 6 + 2]}, {1 / res_params[i * 6 + 3]})")
    print(f"max intensity = {res_params[i * 6 + 4]}")
    print(f"position angle = {res_params[i * 6 + 5]}")
    print(f"flux = {flux((res_params[i * 6 + 2], res_params[i * 6 + 3]), res_params[i * 6 + 4])}")
