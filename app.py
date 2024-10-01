import logging
import math
import os

from dotenv import load_dotenv
import numpy as np
import openai
from PIL import Image
import plotly.graph_objects as go
from scipy.constants import pi as PI
from scipy.stats import invgauss, norm, randint, uniform
import streamlit as st
import torch
import umap
import vec2text

load_dotenv()

# Configure root logger to capture only WARN or higher level logs
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Configure logger to capture DEBUG-level logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

favicon = Image.open("images/favicon.ico")

# Set page layout
st.set_page_config(layout="wide", page_title="Dataset Generator", page_icon=favicon)
st.logo("images/thoughworks_logo.png")

st.title('🖨️ Dataset Generator')
st.markdown("""
            This demo is a practical example of the geometric approach to latent space sampling as described in the paper [Navigating the Geometry of Language: A New Approach to Synthetic Text Generation](https://www.watchful.io/blog/navigating-the-geometry-of-language-a-new-approach-to-synthetic-text-generation). It allows you to generate new synthetic data given some reference text using OpenAI’s ada-002 embedding model. You can browse the source on [GitHub](https://github.com/Watchfulio/dataset-generator).
            """)

st.divider()

if os.environ.get('OPENAI_API_KEY') is None:
    st.error("Please set your OpenAI API key as an environment variable.", icon="🛑")

MODEL = "text-embedding-ada-002"

@st.cache_data(show_spinner=False)
def get_embeddings_openai(text_list: list[str]) -> torch.Tensor:
    """
    Fetch the embeddings from OpenAI and return a tensor

    Args:
      text_list: A list of strings from the user

    Returns:
      torch.Tensor: A tensor of embeddings
    """
    api_key = os.getenv("OPENAI_API_KEY", "Not Found")
    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(
            input=text_list,
            model=MODEL,
            encoding_format="float",
        )
    embeddings = [e.embedding for e in response.data]

    return torch.tensor(embeddings)

@st.cache_resource(show_spinner=False)
def load_corrector() -> vec2text.Corrector:
    corrector = vec2text.load_pretrained_corrector(MODEL)

    return corrector


def calculate_centroid(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Calculate the centroid from a set of embeddings.

    Args:
    embeddings: The embeddings as torch.Tensor.

    Returns:
    torch.Tensor: The centroid as torch.Tensor.
    """
    centroid = embeddings.mean(dim=0)

    return centroid

def calculate_cone_height(embeddings_tensor: torch.Tensor, centroid: torch.Tensor, percentile: int=99):
    """
    Use the centroid and a percentile to calculate the height of the cone.

    Args:
        embeddings_tensor: The embeddings as torch.Tensor.
        centroid: The centroid as torch.Tensor.
        percentile: The percentile to use when calculating the cone height.

    Returns:
        torch.Tensor: The height of the cone.
    """
    # Calculate the centered projections of the embedding vectors onto the centroid vector.
    centroid_norm = torch.norm(centroid)
    projections = torch.matmul(embeddings_tensor, centroid) / centroid_norm
    centered_projections = torch.abs(projections - centroid_norm)

    # Set cone height as the centered_projection at a desired percentile.
    cone_height = torch.quantile(centered_projections, percentile / 100.0)

    return cone_height

def calculate_cone_angle(embeddings_tensor: torch.Tensor, centroid: torch.Tensor, height: float, percentile: int=99):
    """
    From the centroid, cone height and percentile, calculate the cone angle.

    Args:
        embeddings_tensor: The embeddings as torch.Tensor.
        centroid: The centroid as torch.Tensor.
        height: The height of the cone.
        percentile: The percentile to use when calculating the cone angle.

    Returns:
        float: The angle of the cone.
    """
    # Calculate angles between the embedding vectors and the vertex vector.
    vertex = centroid + height * centroid / torch.norm(centroid)
    embeddings_to_vertex = vertex - embeddings_tensor
    dot_products = torch.matmul(embeddings_to_vertex, vertex)
    norms = torch.norm(embeddings_to_vertex, dim=1) * torch.norm(vertex)
    cos_angles = dot_products / norms
    angles = torch.abs(torch.acos(cos_angles))  # Angles in radians.

    # Set the cone angle as the angle at a desired percentile.
    cone_angle_radians = torch.quantile(torch.stack([angles, PI/2 - angles]), percentile / 100.0)

    return cone_angle_radians


def sample_hypersphere(centroid: torch.Tensor, radius: float, m: int, n: int, distribution: str ="normal", use_gpu: bool=False):
    """
    Samples m points within an n-dimensional hypersphere.

    Args:
        centroid (torch.Tensor): The centroid of the hypersphere, a vector of size n.
        radius (float): The radius of the hypersphere.
        m (int): The number of points to sample.
        n (int): The dimension of the space.
        distribution (str): The distribution to be sampled - "normal", "uniform", "inverse_normal"
        use_gpu (bool): Whether to use the GPU in calculating the samples or not.

    Returns:
        torch.Tensor: m points within the n-dimensional hypersphere.
    """
    assert distribution in ["uniform", "normal", "inverse_normal"]
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    centroid = centroid.to(device)
    # Generate m points from a Gaussian distribution
    points = torch.randn(m, n, device=device)

    if distribution == "normal":
        # Calculate the scaling factor to ensure all points are within the radius
        max_distance = points.norm(dim=1).max()
        scaling_factor = radius / max_distance

        # Scale the points
        points = points * scaling_factor

    elif distribution in ["uniform", "surface", "inverse_normal"]:
        # Normalize each point to lie on the surface of a unit hypersphere
        points = points / points.norm(dim=1, keepdim=True)

        # Scale points within the hypersphere (for UNIFORM mode)
        if distribution == "uniform":
            new_radii = torch.pow(torch.rand(m, 1), 1/n)
            points = points * new_radii

        if distribution == "inverse_normal":
            gaussian_noise = torch.abs(torch.randn(m, 1, device=device))
            decay_factor = torch.exp(-gaussian_noise**2 / 8)
            points = points * decay_factor

        # Scale by the radius
        points = points * radius

    # Center points at centroid
    points = points + centroid

    return points

RANDSIGN = randint(low=0, high=2)
UNIFORM = uniform(loc=0, scale=1)
NORMAL = norm(loc=0, scale=1)

def sample_from_spherical_cone(
    cone_centroid: tuple=(0,0,1),
    cone_angle_radians: float=PI/3,
    cone_height: float=1.0,
    distribution: str="normal",
    size: int=100000
) -> np.ndarray:
    """
    Sample from a double spherical cone with the axis derived from the centroid norm.
    The cone centroid, angle, height, distribution and sample size are specifiable.

    Args:
        cone_centroid (tuple): The centroid of the cone.
        cone_angle_radians (float): The angle of the cone in radians.
        cone_height (float): The height of the cone.
        distribution (str): The distribution to be sampled - "normal", "uniform", "inverse_normal"
        size (int): The number of points to sample.

    Returns:
        np.ndarray: The sampled points.
    """
    assert distribution in ["uniform", "normal", "inverse_normal"]
    n_dim = len(cone_centroid)

    direction = (2 * RANDSIGN.rvs(size=size) - 1)
    height = cone_height * UNIFORM.rvs(size=size)**(1/3) * direction
    if distribution == "uniform":
        radius = np.abs(height) * np.tan(cone_angle_radians) * UNIFORM.rvs(size=size)**(1/2)
    elif distribution == "normal":
        rvs = NORMAL.ppf(UNIFORM.rvs(size=size)**(1/2))
        rvs /= rvs.max()
        radius = np.abs(height) * np.tan(cone_angle_radians) * np.abs(rvs)
    elif distribution == "inverse_normal":
        rvs = invgauss.ppf(UNIFORM.rvs(size=size)**(1/2), 0.05)
        rvs /= rvs.max()
        radius = np.abs(height) * np.tan(cone_angle_radians) * np.abs(rvs)

    # Sample points on an (n-1)-dimensional hypersphere
    sphere_points = np.random.normal(size=(size, n_dim-1))
    sphere_points /= np.linalg.norm(sphere_points, axis=1).reshape(-1, 1)

    # Scale points to within the radius
    scaled_points = sphere_points * radius.reshape(-1, 1)

    # Construct points by adding their final coordinates
    final_coord = height - direction * cone_height
    points = np.hstack([scaled_points, final_coord.reshape(-1, 1)])

    # Given unit cone axis
    cone_centroid_norm = cone_centroid / np.linalg.norm(cone_centroid)
    # Original unit cone axis (pointing up in the last dimension)
    original_axis = np.zeros(n_dim)
    original_axis[-1] = 1
    # Apply rotation if the original axis and given axis are not nearly parallel or antiparallel
    if not np.isclose(np.abs(np.dot(original_axis, cone_centroid_norm)), 1):
        print("rotating cone axis")
        u = original_axis.reshape(-1,1)
        v = cone_centroid_norm.reshape(-1,1)
        w = u + v
        w /= np.linalg.norm(w)
        r = np.identity(len(cone_centroid_norm)) + 2 * (v @ u.T - w @ w.T)
        points = points @ r.T

    # Translate all points to align the base of the cone with the centroid
    final_points = points + np.array(cone_centroid).reshape(1, -1)

    return final_points


def get_centroid_and_text(
    embeddings: torch.Tensor,
    iterations: int,
    corrector: vec2text.Corrector
) -> tuple[torch.Tensor, list[str]]:
    """
    Calculate the centroid and text for the centroid.

    Args:
        embeddings: The embeddings as torch.Tensor.
        iterations: The number of iterations to use when inverting the embeddings.
        corrector: The corrector to use when inverting the embeddings.

    Returns:
        tuple[torch.Tensor, list[str]]: The centroid and text for the centroid.
    """
    centroid = calculate_centroid(embeddings)
    if torch.cuda.is_available():
        centroid_text = vec2text.invert_embeddings(embeddings=centroid.unsqueeze(0).cuda(),
                                                    corrector=corrector,
                                                    num_steps=iterations)
    else:
        centroid_text = vec2text.invert_embeddings(embeddings=centroid.unsqueeze(0).to(torch.device("cpu")),
                                                corrector=corrector,
                                                num_steps=iterations)

    return centroid, centroid_text

def get_sphere_samples(
    centroid: torch.Tensor,
    example_count: int,
    distribution: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get samples from a hypersphere.

    Args:
        centroid: The centroid as torch.Tensor.
        example_count: The number of examples to generate.
        distribution: The distribution to use when sampling.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The sampled embeddings and the space embeddings.
    """
    sampled_embeddings = sample_hypersphere(centroid, 0.2, example_count, centroid.shape[1], distribution=distribution, use_gpu=False)
    space_embeddings = sample_hypersphere(centroid, 0.2, 10000, centroid.shape[1], distribution=distribution, use_gpu=False)

    return sampled_embeddings, space_embeddings

def get_cone_samples(
    embeddings: torch.Tensor,
    centroid: torch.Tensor,
    example_count: int,
    percentile: int,
    distribution: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get samples from a cone.

    Args:
        embeddings: The embeddings as torch.Tensor.
        centroid: The centroid as torch.Tensor.
        example_count: The number of examples to generate.
        percentile: The percentile to use when calculating the cone height.
        distribution: The distribution to use when sampling.

    Returns:
        tuple[np.ndarray, np.ndarray]: The sampled embeddings and the space embeddings.
    """
    if st.session_state.calculate_cone_dims is True:
        if torch.cuda.is_available():
            cone_angle_radians = calculate_cone_angle(embeddings.cuda(), centroid.cuda(), torch.Tensor([cone_height]).cuda(), percentile).cpu().item()
            cone_height = calculate_cone_height(embeddings.cuda(), centroid.cuda(), percentile).cpu().item()
        else:
            cone_height = calculate_cone_height(embeddings.to(torch.device("cpu")), centroid.to(torch.device("cpu")), percentile).to(torch.device("cpu")).item()
            cone_angle_radians = calculate_cone_angle(embeddings.to(torch.device("cpu")), centroid.to(torch.device("cpu")), torch.Tensor([cone_height]).to(torch.device("cpu")), percentile).to(torch.device("cpu")).item()

        st.session_state.cone_angle_degrees = math.degrees(cone_angle_radians)
        st.session_state.cone_height = cone_height
    else:
        cone_angle_radians = math.radians(st.session_state.cone_angle_degrees)
        cone_height = st.session_state.cone_height

    sampled_embeddings = sample_from_spherical_cone(centroid.cpu().numpy(), cone_angle_radians, cone_height, distribution=distribution, size=example_count)
    space_embeddings = sample_from_spherical_cone(centroid.cpu().numpy(), cone_angle_radians, cone_height, distribution=distribution, size=10000)

    return sampled_embeddings, space_embeddings


def plot_full_distribution(
    space_embeddings_reduced: np.ndarray,
    centroid_reduced: np.ndarray,
    sampled_embeddings_reduced: np.ndarray,
    provided_embeddings_reduced: np.ndarray,
    centroid_text=None,
    sampled_text=None,
    provided_text=None
) -> go.Figure:
    """
    Plot the full distribution of the embeddings.

    Args:
        space_embeddings_reduced: The reduced space embeddings.
        centroid_reduced: The reduced centroid embeddings.
        sampled_embeddings_reduced: The reduced sampled embeddings.
        provided_embeddings_reduced: The reduced provided embeddings.
        centroid_text: The centroid text.
        sampled_text: The sampled text.
        provided_text: The provided text.

    Returns:
        go.Figure: The plotly figure.
    """
    # Scatter plot for space embeddings
    space_scatter = go.Scatter3d(
        x=space_embeddings_reduced[:, 0],
        y=space_embeddings_reduced[:, 1],
        z=space_embeddings_reduced[:, 2],
        mode='markers',
        marker=dict(size=1, opacity=0.2),
        name='Space'
    )

    # Scatter plot for centroid
    centroid_scatter = go.Scatter3d(
        x=[centroid_reduced[0, 0]],
        y=[centroid_reduced[0, 1]],
        z=[centroid_reduced[0, 2]],
        mode='markers',
        marker=dict(color='red', size=5),
        text=[centroid_text] if centroid_text else None,
        hoverinfo='text' if centroid_text else None,
        name='Centroid'
    )

    # Scatter plot for sampled embeddings
    sampled_scatter = go.Scatter3d(
        x=sampled_embeddings_reduced[:, 0],
        y=sampled_embeddings_reduced[:, 1],
        z=sampled_embeddings_reduced[:, 2],
        mode='markers',
        marker=dict(color='blue', size=3, opacity=0.5),
        text=sampled_text,
        hoverinfo='text' if sampled_text else None,
        name='Sampled'
    )

    # Scatter plot for provided embeddings
    provided_scatter = go.Scatter3d(
        x=provided_embeddings_reduced[:, 0],
        y=provided_embeddings_reduced[:, 1],
        z=provided_embeddings_reduced[:, 2],
        mode='markers',
        marker=dict(color='green', size=3, opacity=0.8),
        text=provided_text,
        hoverinfo='text' if provided_text else None,
        name='Provided'
    )

    # Setting up the layout
    layout = go.Layout(
        scene=dict(
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            zaxis_title="UMAP3"
        ),
        margin=dict(r=0, b=0, l=0, t=0),  # Tight layout
        showlegend=True
    )

    # Creating the figure and adding the scatter plots
    fig = go.Figure(data=[space_scatter, centroid_scatter, sampled_scatter, provided_scatter], layout=layout)

    # Showing the figure
    return fig

####################################################################################################
# Begin Streamlit configuration
####################################################################################################

if "calculate_cone_dims" not in st.session_state:
    st.session_state.calculate_cone_dims = True

if "example_count" not in st.session_state:
    st.session_state.example_count = 10

if "iterations" not in st.session_state:
    st.session_state.iterations = 1

if "cone_angle_degrees" not in st.session_state:
    st.session_state.cone_angle_degrees = math.degrees(PI/2)

if "cone_height" not in st.session_state:
    st.session_state.cone_height = 0.02

if "sample_shape" not in st.session_state:
    st.session_state.sample_shape = "cone"

if "percentile_covered" not in st.session_state:
    st.session_state.percentile_covered = 99

if "distribution" not in st.session_state:
    st.session_state.distribution = "normal"

if "is_first_run" not in st.session_state:
    st.session_state.is_first_run = True


col1, col2 = st.columns(2, gap="medium")
with col1:
    with st.form("sample_parameters", border=False):
        # First row, 2 columns
        r1c1, r1c2 = st.columns(2, gap="medium")
        with r1c1:
            example_count = st.number_input(
                'Examples to generate',
                min_value=5, max_value=100, step=5,
                key="example_count",
                help="How many examples should the generator create for you?"
            )

        with r1c2:
            iterations = st.number_input(
                'Decoding iterations',
                min_value=1, max_value=5, step=1,
                key="iterations",
                help="The embedding inversion is an iterative process. How many iterations should be used?"
            )

        # Second row, 2 columns
        r2c1, r2c2, r2c3 = st.columns(3, gap="medium")
        with r2c1:
            sample_shape = st.selectbox(
                'Sample shape',
                ['cone', 'sphere'],
                key="sample_shape",
                help="What shape should be used to model the space?",
            )

        with r2c2:
            distribution = st.selectbox(
                'Sample distribution',
                ["normal", "uniform", "inverse_normal"],
                key="distribution",
                help="What distribution method should be used to sample the space?"
            )

        with r2c3:
            percentile_covered = st.number_input(
                'Percentile covered',
                min_value=5, max_value=100, step=5,
                key="percentile_covered",
                help="What percentile of the cone should be covered by the samples?"
            )

        # 3rd row, 2 columns
        r3c1, r3c2 = st.columns(2, gap="medium")
        with r3c1:
            if st.session_state.calculate_cone_dims is True:
                angle = st.empty()
            else:
                angle = st.number_input(
                    'Cone angle in degrees',
                    min_value=1.0, max_value=90.0, step=2.0,
                    key="cone_angle_degrees",
                    help="What should the cone angle be? A larger angle implies a flattened, fatter cone."
                )

        with r3c2:
            if st.session_state.calculate_cone_dims is True:
                height = st.empty()
            else:
                height = st.number_input(
                    'Cone height',
                    min_value=0.01, max_value=1.0, step=0.01,
                    key="cone_height",
                    help="What should the cone height be? A larger height implies a taller, thinner cone."
                )

        example_text = st.text_area(
            'Reference text to be used to generate more. One example per line. Plain text only.',
            height=440, max_chars=7000, key="example_text",
        )

        submitted = st.form_submit_button('Submit', type="primary")

with col2:
    chart_placeholder = st.empty()

st.divider()

progress = st.empty()

text_placeholder = st.empty()

if submitted and not example_text:
    st.error("Please enter some reference text to be used to generate more.")
    st.stop()

if submitted and os.environ.get('OPENAI_API_KEY') is not None:
    progress.progress(5, "Loading corrector...")
    corrector = load_corrector()

    try:
        with chart_placeholder.container():
            # Get the embeddings for the input text from OpenAI
            progress.progress(10, "Fetching embeddings...")
            embeddings = get_embeddings_openai(example_text.splitlines())

            progress.progress(20, "Mapping embedding space...")
            sample_shape = st.session_state.sample_shape if st.session_state.sample_shape else "cone"
            percentile = st.session_state.percentile_covered if st.session_state.percentile_covered else 99
            distribution = st.session_state.distribution if st.session_state.distribution else "uniform"

            progress.progress(25, "Calculating centroid and text...")
            centroid, centroid_text = get_centroid_and_text(embeddings, iterations, corrector)

            progress.progress(40, "Sampling embeddings...")
            if sample_shape == "sphere":
                st.session_state.calculate_cone_dims = False
                sampled_embeddings, space_embeddings = get_sphere_samples(centroid, example_count, distribution)
                if torch.cuda.is_available():
                    new_embeddings = sampled_embeddings.cuda()
                else:
                    new_embeddings = sampled_embeddings.to(torch.device("cpu"))
            elif sample_shape == "cone":
                sampled_embeddings, space_embeddings = get_cone_samples(embeddings, centroid, example_count, percentile, distribution)
                if torch.cuda.is_available():
                    new_embeddings = torch.from_numpy(sampled_embeddings).float().cuda()
                else:
                    new_embeddings = torch.from_numpy(sampled_embeddings).float().to(torch.device("cpu"))

            progress.progress(50, "Reducing dimensions...")
            # Combine all embeddings
            all_embeddings = np.vstack([centroid, sampled_embeddings, embeddings.cpu().numpy(), space_embeddings])

            # Perform umap to reduce dimensions to 3
            reducer = umap.UMAP(n_neighbors=30, n_components=3, metric='euclidean')
            all_embeddings_reduced = reducer.fit_transform(all_embeddings)

            progress.progress(75, "Inverting embeddings... This may take a while")
            generated_text = vec2text.invert_embeddings(embeddings=new_embeddings,
                                                            corrector=corrector,
                                                            num_steps=iterations)

            progress.progress(90, "Plotting...")
            # Extract the reduced embeddings
            centroid_reduced = all_embeddings_reduced[0:1, :]
            sampled_embeddings_reduced = all_embeddings_reduced[1:1+len(sampled_embeddings), :]
            provided_embeddings_reduced = all_embeddings_reduced[1+len(sampled_embeddings):1+len(sampled_embeddings)+len(embeddings.cpu().numpy()), :]
            space_embeddings_reduced = all_embeddings_reduced[1+len(sampled_embeddings)+len(embeddings.cpu().numpy()):, :]

            fig = plot_full_distribution(
                space_embeddings_reduced,
                centroid_reduced,
                sampled_embeddings_reduced,
                provided_embeddings_reduced,
                centroid_text=centroid_text,
                sampled_text=generated_text,
                provided_text=example_text.splitlines()
            )

            # Regular streamlit chart
            st.plotly_chart(
                fig, theme="streamlit", use_container_width=True,
                key="scatter_chart"
            )

            # If we first calculated the cone dimensions, then we need to display them
            # so the user can see what the calculated values are and adjust them if needed.
            if st.session_state.calculate_cone_dims is True:
                angle.number_input(
                    'Cone angle in degrees',
                    min_value=1.0, max_value=90.0, step=2.0,
                    key="cone_angle_degrees",
                    help="What should the cone angle be? A larger angle implies a flattened, fatter cone."
                )

                height.number_input(
                    'Cone height',
                    min_value=0.01, max_value=1.0, step=0.01,
                    key="cone_height",
                    help="What should the cone height be? A larger height implies a taller, thinner cone."
                )


            if st.session_state.sample_shape == "cone":
                c1, c2 = st.columns(2, gap="large")
                with c1:
                    st.write(f"**Cone Angle:** {round(st.session_state.cone_angle_degrees, 4)} degrees")

                with c2:
                    st.write(f"**Cone Height:** {round(st.session_state.cone_height, 4)}")

                st.session_state.calculate_cone_dims = False

            st.write("**Centroid text:**")
            st.write(centroid_text[0])

            progress.empty()


        with text_placeholder.container(border=False):
            st.subheader('Generated text')

            for text in generated_text:
                st.write(text)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        progress.empty()
        logger.debug(f"EXAMPLE_TEXT: {example_text}")
        logger.exception(e)
