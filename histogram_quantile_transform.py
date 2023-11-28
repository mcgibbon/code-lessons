import numpy as np
from scipy.interpolate import interp1d


def transform(
    data: np.ndarray,
    hist_source: np.ndarray,
    hist_dest: np.ndarray,
    bin_edges: np.ndarray,
):
    """
    Take an array of data from the hist_source distribution and quantile transform it
    to the hist_dest distribution using linear interpolation of the CDF
    between bin edges.
    """
    # Calculate the cumulative distribution function (CDF)
    # for the source and destination histograms
    cdf_source = np.cumsum(hist_source) / np.sum(hist_source)
    cdf_dest = np.cumsum(hist_dest) / np.sum(hist_dest)

    # Create interpolating functions for the CDFs

    # We need to prepend zero as that's the cumulative density at the initial bin edge.
    # There are no values before that bin edge. If we discarded a bin edge instead
    # to align with the shape of the histogram, we would be removing information
    # vital to the transformation.
    cdf_source_interp = interp1d(
        # prepend zero value to CDF
        bin_edges,
        np.concatenate([[0], cdf_source]),
        bounds_error=True,
    )
    cdf_dest_interp = interp1d(
        np.concatenate([[0], cdf_dest]), bin_edges, bounds_error=True
    )

    # Transform the data using the interpolated CDFs
    cdf_source_data = cdf_source_interp(data)
    data_transformed = cdf_dest_interp(cdf_source_data)
    return data_transformed


def test_transform_gaussian_to_uniform():
    # Generate random Gaussian and uniform data
    gaussian_data = np.random.randn(10000)
    uniform_data = np.random.uniform(size=10000)

    # Create histograms for the Gaussian and uniform data
    hist_source, bin_edges = np.histogram(gaussian_data, bins=50)
    hist_dest, _ = np.histogram(uniform_data, bins=bin_edges)

    # Transform the Gaussian data to match the uniform distribution histogram
    data_transformed = transform(gaussian_data, hist_source, hist_dest, bin_edges)

    # Create a histogram for the transformed data
    hist_transformed, _ = np.histogram(data_transformed, bins=bin_edges)

    # Assert transformed data histogram is similar to the uniform histogram
    assert np.sum(np.abs(hist_transformed - hist_dest)) < 0.05 * np.sum(
        np.abs(hist_source - hist_dest)
    )
