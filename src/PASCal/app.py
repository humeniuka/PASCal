import codecs
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_from_directory

import PASCal.utils
from PASCal import __version__
from PASCal.constants import PERCENT
from PASCal.core import PASCalResults, fit
from PASCal.options import Options, PASCalDataType
from PASCal.plotting import PLOTLY_CONFIG

app = Flask(__name__)


@app.route("/")
def index():
    print("Request for index page received")
    return render_template("index.html", __version__=__version__)


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


def _parse_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parses the string data provided in the web form into 3 arrays for
    x-variable, errors and unit cell parameters given x.

    Returns:
        A tuple of x, x_error, and unit_cell parameters.

    """
    raw_data = request.form.get("data")
    if not raw_data:
        raise RuntimeError("No data provided.")

    data = np.loadtxt(
        (line for line in raw_data.splitlines()),
    )

    x = data[:, 0]
    x_error = data[:, 1]
    unit_cells = data[:, 2:]

    return x, x_error, unit_cells

# --- read temperature and lattice parameters from cif file ---

def read_float(string):
    # remove uncertainty, 393.0(2) -> 393.0
    f = float(string.split("(")[0])
    return f

def parse_cif(cif_content, cif_filename):
    lines = cif_content.split('\n')
    data = {}
    required_keys = [
        "_cell_measurement_temperature",
        "_cell_length_a", "_cell_length_b", "_cell_length_c",
        "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
        "_chemical_formula_sum",
    ]
    # abbreviations for long field names
    short_names = {
        "_cell_measurement_temperature": "T",
        "_cell_length_a": "a",
        "_cell_length_b": "b",
        "_cell_length_c": "c",
        "_cell_angle_alpha": "alpha",
        "_cell_angle_beta": "beta",
        "_cell_angle_gamma": "gamma",
        "_chemical_formula_sum": "formula",
    }
    for line in lines:
        words = line.split()
        for key in required_keys:
            short_name = short_names[key]
            if line.startswith(key):
                if "_cell" in key:
                    data[short_name] = read_float(words[1])
                else:
                    data[short_name] = words[1]

    # Check that all required fields are present
    for key in required_keys:
        if short_names[key] not in data:
            raise RuntimeError(f"Cif file {cif_filename} is missing required field {key}")

    df = pd.DataFrame(data, index=[cif_filename])
    # Bring columns in the order expected by Pascal:
    # T  a b c  alpha beta gamma
    df = df.loc[:, [short_names[key] for key in required_keys]]

    return df


def _parse_data_from_cif_files() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read temperatures and lattice parameters from uploaded .cif files.

    Returns:
        A tuple of T, T_error, and unit_cell parameters.
    """
    dataframes = []
    for uploaded_file in request.files.getlist('upload'):
        cif_filename = uploaded_file.filename
        cif_content = codecs.decode(uploaded_file.read(), encoding="utf-8")
        df_ = parse_cif(cif_content, cif_filename)
        dataframes.append(df_)
    if len(dataframes) < 2:
        raise RuntimeError("You have to upload at least *two* .cif files measured at different temperatures")
    df = pd.concat(dataframes)
    # sort rows by temperature
    df.sort_values("T", inplace=True)

    # Check that all .cif files belong to the same molecule. We look at the '_chemical_formula_sum' field
    # to check that the elemental composition is the same in all .cif files.
    unique_formulae = df["formula"].unique()
    if len(unique_formulae) > 1:
        raise RuntimeError(f".cif files belong to different molecules, formulae = {unique_formulae}")
    df = df.drop("formula", axis=1)

    data = df.to_numpy()
    # temperatures in Kelvin
    T = data[:, 0]
    # latttice parameters, a b c alpha beta gamma
    unit_cells = data[:, 1:]
    # Assume that the error for the temperature measurement is 1 K
    T_error = 1.0 * np.ones_like(T)

    return T, T_error, unit_cells


@app.route("/output", methods=["POST"])
def output():
    try:
        options = Options.from_form(request.form)
    except Exception as exc:
        raise RuntimeError(f"Could not parse options: {request.form}\nException: {exc}")

    if len(request.files) == 0:
        try:
            x, x_errors, unit_cells = _parse_data()
        except Exception as exc:
            raise RuntimeError(
                f"Could not parse data: {request.form.get('data')}\nException: {exc}"
            )
    else:
        x, x_errors, unit_cells = _parse_data_from_cif_files()

    fit_results = fit(x, x_errors, unit_cells, options)

    return _render_results(fit_results)


def _render_results(results: PASCalResults) -> str:
    """Take the results of a PASCal fit and render them as HTML.

    Parameters:
        results: The results of a PASCal fit.

    Returns:
        The rendered HTML to serve.

    """

    if results.options.data_type == PASCalDataType.TEMPERATURE:
        return render_template(
            "temperature.html",
            config=json.dumps(PLOTLY_CONFIG),
            __version__=__version__,
            warning=results.warning,
            PlotStrainJSON=results.plot_strain(return_json=True),
            PlotVolumeJSON=results.plot_volume(return_json=True),
            PlotIndicJSON=results.plot_indicatrix(return_json=True),
            Axes=["X<sub>1</sub>", "X<sub>2</sub>", "X<sub>3</sub>", "V"],
            PrinComp=np.round(results.principal_components, 4),
            MedianPrinAxCryst=PASCal.utils.round_array(
                results.median_principal_axis_crys, 4
            ),
            Vol=PASCal.utils.round_array(results.cell_volumes, 4),
            PrinAxCryst=PASCal.utils.round_array(results.principal_axis_crys, 4),
            TPx=results.x,
            DiagStrain=np.round(results.diagonal_strain * PERCENT, 4),
            TPxError=results.x_errors,
            Latt=results.unit_cells,
            **{
                k: PASCal.utils.round_array(results.named_coefficients[k], 4)
                for k in results.named_coefficients
            },
        )

    elif results.options.data_type == PASCalDataType.PRESSURE:
        return render_template(
            "pressure.html",
            config=json.dumps(PLOTLY_CONFIG),
            warning=results.warning,
            __version__=__version__,
            PlotStrainJSON=results.plot_strain(return_json=True),
            PlotVolumeJSON=results.plot_volume(return_json=True),
            PlotIndicJSON=results.plot_indicatrix(return_json=True),
            PlotDerivJSON=results.plot_compressibility(return_json=True),
            Axes=["X<sub>1</sub>", "X<sub>2</sub>", "X<sub>3</sub>", "V"],
            PrinComp=np.round(results.principal_components, 4),
            MedianPrinAxCryst=PASCal.utils.round_array(
                results.median_principal_axis_crys, 4
            ),
            PrinAxCryst=PASCal.utils.round_array(results.principal_axis_crys, 4),
            BMOrder=["2nd", "3rd"]
            if not (results.options.use_pc)
            else ["2nd", "3rd", "3rd with P<sub>c</sub>"],
            TPxError=results.x_errors,
            u=results.median_x,
            UsePc=results.options.use_pc,
            Latt=results.unit_cells,
            K=PASCal.utils.round_array(results.compressibility, 4),
            KErr=PASCal.utils.round_array(results.compressibility_errors, 4),
            TPx=results.x,
            DiagStrain=np.round(results.diagonal_strain * PERCENT, 4),
            Vol=PASCal.utils.round_array(results.cell_volumes, 4),
            **{
                k: PASCal.utils.round_array(results.named_coefficients[k], 4)
                for k in results.named_coefficients
            },
        )
    elif results.options.data_type == PASCalDataType.ELECTROCHEMICAL:
        return render_template(
            "electrochem.html",
            config=json.dumps(PLOTLY_CONFIG),
            warning=results.warning,
            __version__=__version__,
            PlotStrainJSON=results.plot_strain(return_json=True),
            PlotVolumeJSON=results.plot_volume(return_json=True),
            PlotIndicJSON=results.plot_indicatrix(return_json=True),
            PlotDerivJSON=results.plot_charge_derivative(return_json=True),
            PlotResidualJSON=results.plot_residual(return_json=True),
            Axes=["X<sub>1</sub>", "X<sub>2</sub>", "X<sub>3</sub>", "V"],
            PrinComp=np.round(results.principal_components, 4),
            MedianPrinAxCryst=PASCal.utils.round_array(
                results.median_principal_axis_crys, 4
            ),
            PrinAxCryst=PASCal.utils.round_array(results.principal_axis_crys, 4),
            TPxError=results.x_errors,
            Latt=results.unit_cells,
            TPx=results.x,
            DiagStrain=np.round(results.diagonal_strain * PERCENT, 4),
            Vol=PASCal.utils.round_array(results.cell_volumes, 4),
            **{
                k: PASCal.utils.round_array(results.named_coefficients[k], 4)
                for k in results.named_coefficients
            },
        )


if __name__ == "__main__":
    app.run(debug=True)
