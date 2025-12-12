from typing import Iterable, Iterator, Any
import base64
from io import BytesIO

import numpy as np
from fastembed import LateInteractionTextEmbedding
from PIL import Image
from pdf2image import convert_from_path
from IPython.display import HTML
import plotly.graph_objects as go


def tokenize_late_interaction(
    model: LateInteractionTextEmbedding, text: str, is_doc: bool = True
) -> list[str]:
    """
    Tokenize text using the provided late interaction text model, such as Colbert.

    Args:
        model: The late interaction text embedding model
        text: The text to tokenize
        is_doc: Whether the text is a document (True) or a query (False)

    Returns:
        List of token strings including special tokens
    """
    # Colbert adds a special marker token at the beginning of the text, after [CLS]
    # This token is different for documents and queries
    marker_token_id = (
        model.model.DOCUMENT_MARKER_TOKEN_ID
        if is_doc
        else model.model.QUERY_MARKER_TOKEN_ID
    )

    # Tokenize the text using the model's tokenizer
    text_tokenization = model.model.tokenize([text], is_doc=is_doc)[0]

    # Insert the marker token after [CLS] (position 0)
    token_ids = [
        text_tokenization.ids[0],
        marker_token_id,
    ] + text_tokenization.ids[1:]

    # Filter out tokens in skip_list, but keep padding tokens
    filtered_ids = [
        tid
        for tid in token_ids
        if tid not in model.model.skip_list or tid == model.model.pad_token_id
    ]

    # Convert token IDs to token strings
    return [model.model.tokenizer.id_to_token(tid) for tid in filtered_ids]


def pdf_to_png_screenshots(pdf_path: str) -> Iterator[Image.Image]:
    """
    Convert PDF pages to PNG screenshots.

    Takes a path to a PDF file and yields PIL Image objects for each page,
    suitable for use with vision models like ColPali.

    Args:
        pdf_path: Path to the PDF file to convert

    Yields:
        PIL Image objects, one for each page of the PDF

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If the PDF cannot be processed
    """
    try:
        pages = convert_from_path(pdf_path, dpi=300, fmt="png")
        for page in pages:
            yield page
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    except Exception as e:
        raise Exception(f"Error processing PDF {pdf_path}: {str(e)}")


def generate_diverse_vectors(
    n_vectors: int,
    n_dim: int,
    rng,
    target_sim_range: tuple[int, int] = (0.5, 1.0),
):
    """
    Generate high-dimensional vectors with diverse cosine similarities.

    Creates a set of unit vectors where each vector has a controlled cosine
    similarity to a base vector, using Gram-Schmidt orthogonalization.

    Args:
        n_vectors: Number of vectors to generate
        n_dim: Dimensionality of each vector
        rng: Random number generator (e.g., np.random.default_rng())
        target_sim_range: Tuple of (min_sim, max_sim) for cosine similarity

    Returns:
        Array of shape (n_vectors, n_dim) with diverse similarities
    """
    # Create normalized base vector
    base_vector = rng.normal(0, 1, n_dim)
    base_vector /= np.linalg.norm(base_vector)

    # Start result list with base vector
    vectors = [base_vector]

    # Generate remaining vectors with controlled similarity
    min_sim, max_sim = target_sim_range
    target_similarities = np.linspace(min_sim, max_sim, n_vectors - 1)

    for target_sim in target_similarities:
        # Create random vector and make it orthogonal to base (Gram-Schmidt)
        random_vec = rng.normal(0, 1, n_dim)
        orthogonal_component = (
            random_vec - np.dot(random_vec, base_vector) * base_vector
        )
        orthogonal_component /= np.linalg.norm(orthogonal_component)

        # Blend base and orthogonal components to achieve target cosine similarity
        # Using the formula: cos(θ) = target_sim
        parallel_weight = target_sim
        orthogonal_weight = np.sqrt(1 - target_sim**2)

        new_vector = (
            parallel_weight * base_vector + orthogonal_weight * orthogonal_component
        )
        vectors.append(new_vector / np.linalg.norm(new_vector))

    return np.array(vectors)


def _get_document_identifier(result: dict, field: str) -> str:
    """Extract a unique identifier for a document from a result."""
    # Try to use ID if available, otherwise hash the content
    if hasattr(result, "id"):
        return str(result.id)
    elif isinstance(result, dict) and "id" in result:
        return str(result["id"])
    else:
        # Fall back to hashing the content
        import hashlib

        content = (
            result.get(field, "")
            if isinstance(result, dict)
            else getattr(result, field, "")
        )
        return hashlib.md5(content.encode()).hexdigest()


def _assign_document_colors(all_results: list[dict], field: str) -> dict[str, str]:
    """Assign unique pastel colors to each unique document."""
    # Pastel color palette for good readability
    colors = [
        "#FFE5E5",  # Light pink
        "#E5F5FF",  # Light blue
        "#E5FFE5",  # Light green
        "#FFF5E5",  # Light orange
        "#F5E5FF",  # Light purple
        "#FFFFE5",  # Light yellow
        "#E5FFFF",  # Light cyan
        "#FFE5F5",  # Light magenta
        "#F5FFE5",  # Light lime
        "#FFE5CC",  # Light peach
        "#E5E5FF",  # Light lavender
        "#E5FFF5",  # Light mint
    ]

    unique_docs = {}
    color_index = 0

    for result in all_results:
        doc_id = _get_document_identifier(result, field)
        if doc_id not in unique_docs:
            unique_docs[doc_id] = colors[color_index % len(colors)]
            color_index += 1

    return unique_docs


def _highlight_query_words(text: str, query: str) -> str:
    """Highlight query words in text with case-insensitive matching."""
    if not query:
        return text

    import re

    # Split query into words, removing punctuation
    query_words = re.findall(r"\w+", query.lower())

    if not query_words:
        return text

    # Create pattern for all query words
    pattern = r"\b(" + "|".join(re.escape(word) for word in query_words) + r")\b"

    # Replace matches with highlighted version (case-insensitive)
    highlighted = re.sub(
        pattern,
        r'<mark style="background-color: #FFEB3B; padding: 2px 4px; border-radius: 3px;">\1</mark>',
        text,
        flags=re.IGNORECASE,
    )

    return highlighted


def display_results_side_by_side(
    left_results: list[dict],
    right_results: list[dict],
    left_title: str = "Results A",
    right_title: str = "Results B",
    field: str = "text",
    query: str = None,
) -> HTML:
    """
    Display two lists of search results side-by-side in a two-column layout.

    Args:
        left_results: List of result dictionaries for left column
        right_results: List of result dictionaries for right column
        left_title: Title for the left column
        right_title: Title for the right column
        field: Dictionary field to display from each result
        query: Optional query string to highlight matching words in results

    Returns:
        IPython HTML display object showing results in two columns with:
        - Unique background colors per document for duplicate detection
        - Highlighted query words (if query provided)
        - Duplicate badges for repeated documents
    """
    # Combine all results to assign consistent colors
    all_results = left_results + right_results
    doc_colors = _assign_document_colors(all_results, field)

    # Build position maps for cross-referencing duplicates
    left_positions = {}  # doc_id -> list of positions in left results
    right_positions = {}  # doc_id -> list of positions in right results

    for i, result in enumerate(left_results, start=1):
        doc_id = _get_document_identifier(result, field)
        if doc_id not in left_positions:
            left_positions[doc_id] = []
        left_positions[doc_id].append(i)

    for i, result in enumerate(right_results, start=1):
        doc_id = _get_document_identifier(result, field)
        if doc_id not in right_positions:
            right_positions[doc_id] = []
        right_positions[doc_id].append(i)

    def format_items(results, is_left_column):
        items_html = ""
        for i, result in enumerate(results, start=1):
            doc_id = _get_document_identifier(result, field)
            bg_color = doc_colors[doc_id]

            # Check for duplicates in the opposite column
            opposite_positions = (
                right_positions.get(doc_id, [])
                if is_left_column
                else left_positions.get(doc_id, [])
            )
            has_duplicate = len(opposite_positions) > 0

            # Get text and apply highlighting if query provided
            text = (
                result.get(field, "")
                if isinstance(result, dict)
                else getattr(result, field, "")
            )
            if query:
                text = _highlight_query_words(text, query)

            # Add position badge if document appears in opposite column
            position_badge = ""
            if has_duplicate:
                positions_str = ", ".join(f"#{pos}" for pos in opposite_positions)
                opposite_label = "right" if is_left_column else "left"
                position_badge = f'<span style="display: inline-block; background-color: #FF5722; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.75em; margin-left: 8px; font-weight: bold;">Also in {opposite_label}: {positions_str}</span>'

            items_html += f"""
            <li style='margin-bottom: 15px; background-color: {bg_color}; padding: 10px; border-radius: 5px; border-left: 4px solid {"#FF5722" if has_duplicate else "#ccc"};'>
                {text}{position_badge}
            </li>
            """
        return items_html

    html_content = f"""
    <div style="display: flex; gap: 20px; font-family: sans-serif;">
        <div style="flex: 1; min-width: 0;">
            <h3 style="margin-top: 0;">{left_title}</h3>
            <ol style="padding-left: 20px;">
                {format_items(left_results, is_left_column=True)}
            </ol>
        </div>
        <div style="flex: 1; min-width: 0;">
            <h3 style="margin-top: 0;">{right_title}</h3>
            <ol style="padding-left: 20px;">
                {format_items(right_results, is_left_column=False)}
            </ol>
        </div>
    </div>
    """
    return HTML(html_content)


def batched(iterable: Iterable[Any], n: int) -> Iterable[Any]:
    """
    Batch data into lists of length n. The last batch may be shorter.

    Args:
        iterable: An iterable to batch
        n: Batch size

    Yields:
        Lists of up to n items from the iterable
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def _make_labels_unique(tokens: list[str]) -> list[str]:
    """Make token labels unique by appending position indices to duplicates.

    Args:
        tokens: List of token strings that may contain duplicates

    Returns:
        List of unique labels with position indices appended to duplicates

    Example:
        >>> _make_labels_unique(['a', 'b', 'a', 'c', 'a'])
        ['a [0]', 'b', 'a [2]', 'c', 'a [4]']
    """
    from collections import Counter

    # Count occurrences of each token
    token_counts = Counter(tokens)

    # Track which occurrence number we're at for each duplicate token
    seen = {}
    unique_labels = []

    for i, token in enumerate(tokens):
        if token_counts[token] > 1:
            # This token appears multiple times - add position index
            unique_labels.append(f"{token} [{i}]")
        else:
            # Unique token - keep as is
            unique_labels.append(token)

    return unique_labels


def _filter_tokens_and_matrix(
    similarity_matrix: np.ndarray,
    query_tokens: list[str],
    document_tokens: list[str],
    ignore_tokens: list[str],
) -> tuple[list[str], list[str], np.ndarray]:
    """Filter tokens and similarity matrix by removing ignored tokens."""
    query_mask = np.array([tok not in ignore_tokens for tok in query_tokens])
    doc_mask = np.array([tok not in ignore_tokens for tok in document_tokens])

    filtered_query = [tok for tok in query_tokens if tok not in ignore_tokens]
    filtered_doc = [tok for tok in document_tokens if tok not in ignore_tokens]
    filtered_matrix = similarity_matrix[query_mask][:, doc_mask]

    return filtered_query, filtered_doc, filtered_matrix


def _create_maxsim_annotations(
    max_indices: np.ndarray, matrix_t: np.ndarray, show_values: bool
) -> tuple[list[dict], list[dict]]:
    """Create shapes and annotations for MaxSim highlighting.

    Args:
        max_indices: Index of best doc token for each query token
        matrix_t: Transposed similarity matrix (n_doc x n_query)
        show_values: Whether to show value annotations
    """
    shapes = []
    annotations = []

    for j in range(len(max_indices)):
        i = max_indices[j]  # Best doc token index for this query token

        # Add red border rectangle for this MaxSim cell
        shapes.append(
            dict(
                type="rect",
                x0=j - 0.5,
                y0=i - 0.5,
                x1=j + 0.5,
                y1=i + 0.5,
                line=dict(color="#8B0000", width=6),
            )
        )

        # Optionally add text annotation
        if show_values:
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"<b>{matrix_t[i, j]:.3f}</b>",
                    showarrow=False,
                    font=dict(
                        color="black",
                        size=11,
                        family="Arial, sans-serif",
                    ),
                    bgcolor="rgba(255, 255, 255, 0.85)",
                    borderpad=3,
                    bordercolor="red",
                    borderwidth=1,
                )
            )

    return shapes, annotations


def visualize_maxsim_matrix(
    similarity_matrix: np.ndarray,
    query_tokens: list[str],
    document_tokens: list[str],
    width: int = 800,
    height: int = None,
    ignore_tokens: list[str] = None,
    show_values: bool = False,
) -> go.Figure:
    """
    Visualize a similarity matrix with MaxSim highlighting.

    Creates an interactive heatmap showing token-level similarities between
    query and document tokens. Highlights the maximum similarity values used
    in MaxSim scoring (best document token per query token) with thick red
    borders and optional bold value annotations.

    Args:
        similarity_matrix: Similarity scores of shape (n_query_tokens, n_doc_tokens)
        query_tokens: List of query token strings (displayed on x-axis)
        document_tokens: List of document token strings (displayed on y-axis)
        width: Figure width in pixels
        height: Figure height in pixels. If None, automatically calculated based on token count
        ignore_tokens: List of token strings to filter out (e.g., ["[MASK]", "[PAD]"]).
                      Defaults to ["[MASK]"] to ignore padding tokens.
                      Pass [] to disable filtering.
        show_values: If True, display bold similarity values on MaxSim cells with
                    semi-transparent background for enhanced visibility

    Returns:
        Plotly Figure object with the annotated heatmap

    Example:
        >>> maxsim_fig = visualize_maxsim_matrix(
        ...     similarity_matrix,
        ...     query_tokens,
        ...     document_tokens,
        ...     show_values=True
        ... )
        >>> maxsim_fig.show()
    """
    # Default to ignoring [MASK] tokens
    if ignore_tokens is None:
        ignore_tokens = ["[MASK]"]

    # Filter tokens and matrix
    filtered_query, filtered_doc, filtered_matrix = _filter_tokens_and_matrix(
        similarity_matrix,
        query_tokens,
        document_tokens,
        ignore_tokens,
    )

    # Make labels unique to handle duplicate tokens correctly
    unique_query_labels = _make_labels_unique(filtered_query)
    unique_doc_labels = _make_labels_unique(filtered_doc)

    # Compute MaxSim: for each query token, find max similarity across doc tokens
    max_indices = np.argmax(filtered_matrix, axis=1)  # Best doc token per query

    # Transpose for visualization (doc tokens on y-axis, query tokens on x-axis)
    matrix_t = filtered_matrix.T
    n_doc, n_query = matrix_t.shape

    # Calculate height if not provided
    if height is None:
        cell_height = min(30, width / n_query)
        height = int(cell_height * n_doc + 150)

    # Create heatmap with unique labels
    heatmap_fig = go.Figure(
        go.Heatmap(
            z=matrix_t,
            x=unique_query_labels,
            y=unique_doc_labels,
            colorscale="YlGnBu",
            showscale=True,
            hovertemplate=(
                "Query: %{x}<br>Document: %{y}<br>Similarity: %{z:.3f}<br><extra></extra>"
            ),
            colorbar=dict(title="Similarity"),
        )
    )

    # Add MaxSim highlighting
    shapes, annotations = _create_maxsim_annotations(max_indices, matrix_t, show_values)

    # Update layout
    heatmap_fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        title=dict(
            text=(
                f"Similarity Matrix with MaxSim Highlighting<br>"
                f"Red borders indicate maximum similarity for each query token</sub>"
            ),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        xaxis=dict(title="Query Tokens", side="top", tickangle=-45),
        yaxis=dict(title="Document Tokens", autorange="reversed"),
        width=width,
        height=height,
        font=dict(size=11),
        margin=dict(t=150, l=100, r=100, b=80),
    )

    return heatmap_fig


def visualize_image_patches(
    image: Image.Image,
    processor,
    patch_size: int = 0,
    line_color: str = "red",
    line_width: int = 3,
) -> go.Figure:
    """
    Visualize how an image is divided into patches by ColPali processor.

    Displays the image with grid lines overlaid to show patch boundaries.
    Much faster than creating individual subplots for each patch.

    Args:
        image: PIL Image to visualize
        processor: ColPali processor (BaseVisualRetrieverProcessor subclass)
        patch_size: Patch size for the model (0 for colSmol, specific value for ColPali)
        line_color: Color of the grid lines (default: white)
        line_width: Width of the grid lines in pixels (default: 2)

    Returns:
        Plotly Figure object showing the image with patch grid overlay

    Example:
        >>> from PIL import Image
        >>> image = Image.open("document.png")
        >>> fig = visualize_image_patches(image, processor, patch_size=16)
        >>> fig.show()
    """
    # Resize image to max 600px width while maintaining aspect ratio
    max_width = 600
    if image.width > max_width:
        aspect_ratio = image.height / image.width
        new_width = max_width
        new_height = int(new_width * aspect_ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Get number of patches in each dimension
    n_patches_height, n_patches_width = processor.get_n_patches(
        image_size=image.size, patch_size=patch_size
    )

    # Convert image to numpy array after ensuring RGB format
    img_array = np.array(image.convert("RGB"))
    original_height, original_width = img_array.shape[:2]

    # Calculate patch dimensions
    patch_height = original_height // n_patches_height
    patch_width = original_width // n_patches_width

    # Crop image to show only evenly-divisible patches
    cropped_height = n_patches_height * patch_height
    cropped_width = n_patches_width * patch_width
    img_array = img_array[:cropped_height, :cropped_width]

    # Create figure with the cropped image
    fig = go.Figure(go.Image(z=img_array))

    # Create grid lines as shapes
    shapes = []

    # Vertical lines (including edges at 0 and width)
    for j in range(n_patches_width + 1):
        x_pos = j * patch_width
        shapes.append(
            dict(
                type="line",
                x0=x_pos,
                y0=0,
                x1=x_pos,
                y1=cropped_height,
                line=dict(color=line_color, width=line_width),
            )
        )

    # Horizontal lines (including edges at 0 and height)
    for i in range(n_patches_height + 1):
        y_pos = i * patch_height
        shapes.append(
            dict(
                type="line",
                x0=0,
                y0=y_pos,
                x1=cropped_width,
                y1=y_pos,
                line=dict(color=line_color, width=line_width),
            )
        )

    # Calculate appropriate figure dimensions
    aspect_ratio = cropped_height / cropped_width
    fig_width = image.width
    fig_height = int(fig_width * aspect_ratio) + 100

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Image Divided into {n_patches_height}×{n_patches_width} Patches",
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            range=[0, cropped_width],
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            range=[cropped_height, 0],
        ),
        width=fig_width,
        height=fig_height,
        shapes=shapes,
        showlegend=False,
        margin=dict(l=10, r=10, t=80, b=10),
    )

    return fig


def compare_optimization_methods(
    query: str,
    query_embedding: list,
    client,
    collection_name: str,
    vector_names: list[str],
    limit: int = 3,
    image_height: int = 150,
) -> go.Figure:
    """
    Compare search results across different optimization methods.

    Creates an interactive Plotly visualization with images showing retrieval results
    from multiple optimization strategies. Each method is displayed as a row with its
    top results shown horizontally. The first row shows the baseline method, with
    recall metrics calculated for each subsequent method.

    Args:
        query: The search query string
        query_embedding: The query embedding vector
        client: QdrantClient instance
        collection_name: Name of the Qdrant collection
        vector_names: List of vector names to compare (first one is baseline)
        limit: Number of top results to retrieve per method
        image_height: Height in pixels for each image row

    Returns:
        Plotly Figure object with side-by-side image comparison

    Example:
        >>> fig = compare_optimization_methods(
        ...     query="coffee mug",
        ...     query_embedding=query_embeddings[0],
        ...     client=client,
        ...     collection_name="colpali-optimizations",
        ...     vector_names=["original", "binary_quantized", "hierarchical_2x"],
        ...     limit=3,
        ...     image_height=200
        ... )
        >>> fig.show()
    """
    from qdrant_client import models as qdrant_models

    # Collect results for all methods
    all_results = {}
    for vector_name in vector_names:
        results = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            using=vector_name,
            search_params=qdrant_models.SearchParams(
                quantization=qdrant_models.QuantizationSearchParams(
                    rescore=False,
                )
            ),
            limit=limit,
        )
        all_results[vector_name] = results.points

    # Get baseline (first method) IDs for precision calculation
    baseline_ids = [r.id for r in all_results[vector_names[0]]]

    # Calculate precision for each method
    precisions = {}
    for vector_name in vector_names:
        result_ids = [r.id for r in all_results[vector_name]]
        overlap = len(set(baseline_ids) & set(result_ids))
        precisions[vector_name] = overlap / len(baseline_ids)

    # Create image-based visualization
    from plotly.subplots import make_subplots

    # Create subplots: rows for methods, columns for results
    fig = make_subplots(
        rows=len(vector_names),
        cols=limit,
        vertical_spacing=0.05,
        horizontal_spacing=0.01,
    )

    # Load and add images
    for row_idx, vector_name in enumerate(vector_names, start=1):
        for col_idx, result in enumerate(all_results[vector_name], start=1):
            try:
                img = Image.open(result.payload["image_path"])
                # Resize to manageable size
                aspect_ratio = img.height / img.width
                new_width = 300
                new_height = int(new_width * aspect_ratio)
                img_resized = img.resize(
                    (new_width, new_height),
                    Image.Resampling.LANCZOS,
                )
                img_array = np.array(img_resized.convert("RGB"))

                fig.add_trace(go.Image(z=img_array), row=row_idx, col=col_idx)
            except Exception:
                # If image loading fails, skip
                pass

    # Add row headers with color-coding based on precision
    for i, vector_name in enumerate(vector_names):
        precision = precisions[vector_name]
        precision_pct = precisions[vector_name] * 100

        # Determine background color based on precision
        if precision >= 0.9:
            bgcolor = "#C8E6C9"  # Light green
        elif precision >= 0.6:
            bgcolor = "#FFF9C4"  # Light yellow
        else:
            bgcolor = "#FFCDD2"  # Light red

        # Create label text
        if vector_name == vector_names[0]:
            label_text = f"<b>{vector_name}</b><br>(baseline)"
        else:
            label_text = f"<b>{vector_name}</b><br>(precision: {precision_pct:.0f}%)"

        # Calculate y position for this row (centered vertically in the row)
        # Rows are distributed from 1 (top) to 0 (bottom) in normalized coordinates
        row_fraction = 1 / len(vector_names)
        y_pos = 1 - (i + 0.5) * row_fraction

        fig.add_annotation(
            text=label_text,
            xref="paper",
            yref="paper",
            x=-0.05,  # Position to the left of the plot area
            y=y_pos,
            xanchor="right",
            yanchor="middle",
            showarrow=False,
            bgcolor=bgcolor,
            borderpad=6,
            font=dict(size=10),
        )

    # Hide axes
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)

    # Update layout
    total_height = len(vector_names) * image_height + 150
    fig.update_layout(
        title=dict(
            text=f'<b>Optimization Methods Comparison</b><br><sub>Query: "{query}"</sub>',
            x=0.5,
            xanchor="center",
        ),
        width=limit * 300,
        height=total_height,
        showlegend=False,
        margin=dict(l=150, r=10, t=120, b=10),
    )

    return fig


def display_search_results(
    results: list,
    layout: str = "vertical",
    max_width: int = 400,
    show_scores: bool = True,
    height_per_image: int = 500,
) -> HTML:
    """
    Display search results with properly scaled images.

    Takes a list of search results containing image paths and displays them
    in a readable format with optional similarity scores. Uses IPython HTML
    display for crisp image rendering without blurring.

    Args:
        results: List of search result objects with .payload["file_path"] and optional .score
        layout: "vertical" for stacked layout (default), "horizontal" for side-by-side
        max_width: Maximum width in pixels for each image (maintains aspect ratio)
        show_scores: If True, display similarity scores as subtitles
        height_per_image: Height allocation per image in vertical layout (unused, kept for compatibility)

    Returns:
        IPython HTML display object with the image display

    Example:
        >>> from qdrant_client import QdrantClient
        >>> client = QdrantClient("http://localhost:6333")
        >>> search_results = client.query_points("collection", query=[0.1]*128)
        >>> display_html = display_search_results(search_results.points, show_scores=True)
        >>> display(display_html)
    """
    n_results = len(results)

    # Extract file paths and scores
    file_paths = [point.payload["file_path"] for point in results]
    scores = [point.score if hasattr(point, "score") else None for point in results]

    # Build HTML based on layout
    if layout == "vertical":
        html_content = '<div style="font-family: sans-serif;">'

        for i, (file_path, score) in enumerate(zip(file_paths, scores)):
            # Load and resize image
            img = Image.open(file_path)
            aspect_ratio = img.height / img.width
            new_width = min(max_width, img.width)
            new_height = int(new_width * aspect_ratio)
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to base64
            img_base64 = pil_image_to_base64(img_resized, format="PNG")

            # Build item HTML
            score_text = (
                f"<h4 style='margin: 10px 0 5px 0;'>Score: {score:.4f}</h4>"
                if score is not None and show_scores
                else ""
            )

            html_content += f"""
            <div style="margin-bottom: 20px;">
                {score_text}
                <img src="data:image/png;base64,{img_base64}" style="max-width: {max_width}px; border: 1px solid #ddd; border-radius: 4px;">
            </div>
            """

        html_content += "</div>"

    else:  # horizontal layout
        html_content = '<div style="display: flex; gap: 15px; font-family: sans-serif; overflow-x: auto;">'

        for i, (file_path, score) in enumerate(zip(file_paths, scores)):
            # Load and resize image
            img = Image.open(file_path)
            aspect_ratio = img.height / img.width
            new_width = min(max_width, img.width)
            new_height = int(new_width * aspect_ratio)
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to base64
            img_base64 = pil_image_to_base64(img_resized, format="PNG")

            # Build item HTML
            score_text = (
                f"<h4 style='margin: 0 0 8px 0; text-align: center;'>Score: {score:.4f}</h4>"
                if score is not None and show_scores
                else ""
            )

            html_content += f"""
            <div style="flex-shrink: 0;">
                {score_text}
                <img src="data:image/png;base64,{img_base64}" style="max-width: {max_width}px; border: 1px solid #ddd; border-radius: 4px;">
            </div>
            """

        html_content += "</div>"

    return HTML(html_content)


def visualize_simhash_boundaries(
    simhash_coeffs: np.ndarray,
    x_range: tuple[float, float] = (-1, 1),
    y_range: tuple[float, float] = (-1, 1),
    width: int = 600,
    height: int = 600,
    title: str = "SimHash Decision Boundaries",
) -> go.Figure:
    """
    Visualize SimHash hyperplane decision boundaries in 2D space.

    Displays the hyperplanes that partition the 2D vector space into clusters.
    Each line represents a decision boundary where points on opposite sides
    belong to different clusters.

    Args:
        simhash_coeffs: SimHash coefficient matrix of shape (2, k_sim)
        x_range: Tuple of (min, max) for x-axis range
        y_range: Tuple of (min, max) for y-axis range
        width: Figure width in pixels
        height: Figure height in pixels
        title: Plot title

    Returns:
        Plotly Figure object showing the decision boundaries

    Example:
        >>> from fastembed.postprocess.muvera import SimHashProjection
        >>> simhash = SimHashProjection(k_sim=3, dim=2, random_generator=rng)
        >>> fig = visualize_simhash_boundaries(simhash.simhash_vectors)
        >>> fig.show()
    """
    fig = go.Figure()

    # Plot the decision boundaries (hyperplanes) for each simhash vector
    x_vals = np.linspace(x_range[0], x_range[1], 100)
    for i, c in enumerate(simhash_coeffs.T):
        # For a 2D hyperplane ax + by = 0, we solve for y = -ax/b
        if abs(c[1]) > 1e-10:  # Avoid division by zero
            y_boundary = -c[0] * x_vals / c[1]
            # Only plot within reasonable bounds
            mask = (y_boundary >= y_range[0]) & (y_boundary <= y_range[1])
            fig.add_trace(
                go.Scatter(
                    x=x_vals[mask],
                    y=y_boundary[mask],
                    mode="lines",
                    name=f"Boundary {i + 1}",
                    line=dict(width=2),
                )
            )
        else:  # Vertical line case
            fig.add_trace(
                go.Scatter(
                    x=[0, 0],
                    y=[y_range[0], y_range[1]],
                    mode="lines",
                    name=f"Boundary {i + 1}",
                    line=dict(width=2),
                )
            )

    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        xaxis_range=[x_range[0], x_range[1]],
        yaxis_range=[y_range[0], y_range[1]],
        title=title,
    )
    return fig


def visualize_random_projection_quality(
    original_vectors: np.ndarray,
    original_similarity: np.ndarray,
    dim_range: np.ndarray,
    width: int = 900,
    height: int = 500,
    title: str = "Random Projection: Compression vs Accuracy",
) -> go.Figure:
    """
    Visualize Random Projection quality across different dimensionalities.

    Compares original similarity matrix against Random Projection compressed
    versions at various target dimensions, showing the compression-accuracy tradeoff.

    Args:
        original_vectors: Original high-dimensional vectors of shape (n_vectors, n_dims)
        original_similarity: Original similarity matrix of shape (n_vectors, n_vectors)
        dim_range: Array of target dimensions to test
        width: Figure width in pixels
        height: Figure height in pixels
        title: Plot title

    Returns:
        Plotly Figure object showing mean error and standard deviation vs dimensions

    Example:
        >>> from sklearn.metrics.pairwise import cosine_similarity
        >>> vectors = generate_diverse_vectors(10, 10000, rng)
        >>> sim_matrix = cosine_similarity(vectors, vectors)
        >>> dims = np.linspace(5000, 1, 50).astype(int)
        >>> fig = visualize_random_projection_quality(vectors, sim_matrix, dims)
        >>> fig.show()
    """
    from sklearn.metrics.pairwise import cosine_similarity

    def random_projection(vectors: np.ndarray, output_dim: int) -> np.ndarray:
        assert len(vectors.shape) == 2, "Expected 2D numpy array"
        n_samples, input_dim = vectors.shape
        projections = np.random.choice([-1, 1], size=(input_dim, output_dim))
        return (1 / np.sqrt(output_dim)) * vectors @ projections

    # Test compression across dimensionalities
    mean_values, std_values = [], []
    for dim in dim_range:
        projected_vectors = random_projection(original_vectors, dim)
        projected_sim = cosine_similarity(projected_vectors, projected_vectors)
        sim_diff = np.abs(projected_sim - original_similarity)
        mean_values.append(sim_diff.mean())
        std_values.append(sim_diff.std())

    # Create error plot with error bars
    fig = go.Figure(
        data=go.Scatter(
            x=dim_range,
            y=mean_values,
            error_y=dict(type="data", array=std_values, visible=True),
            mode="lines+markers",
            name="Mean Error",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Output Dimensions",
        yaxis_title="Mean Similarity Error",
        width=width,
        height=height,
    )
    return fig


def visualize_cluster_distribution(
    doc_cluster_ids: np.ndarray,
    query_cluster_ids: np.ndarray,
    n_clusters: int = 8,
    width: int = 700,
    height: int = 400,
    title: str = "Token Distribution Across Clusters",
) -> go.Figure:
    """
    Visualize token distribution across clusters for document and query.

    Creates a grouped bar chart comparing how tokens are distributed across
    clusters for both document and query, useful for understanding MUVERA's
    clustering behavior.

    Args:
        doc_cluster_ids: Document token cluster assignments
        query_cluster_ids: Query token cluster assignments
        n_clusters: Total number of clusters (default 8)
        width: Figure width in pixels
        height: Figure height in pixels
        title: Plot title

    Returns:
        Plotly Figure object with grouped bar chart

    Example:
        >>> doc_clusters = simhash.get_cluster_ids(doc_embeddings)
        >>> query_clusters = simhash.get_cluster_ids(query_embeddings)
        >>> fig = visualize_cluster_distribution(doc_clusters, query_clusters)
        >>> fig.show()
    """
    from collections import Counter

    # Count tokens per cluster
    doc_cluster_counts = Counter(doc_cluster_ids)
    query_cluster_counts = Counter(query_cluster_ids)

    # Create bar chart comparing distributions
    fig = go.Figure()

    # Add document bars
    fig.add_trace(
        go.Bar(
            x=list(range(n_clusters)),
            y=[doc_cluster_counts.get(i, 0) for i in range(n_clusters)],
            name="Document",
            marker_color="lightblue",
        )
    )

    # Add query bars
    fig.add_trace(
        go.Bar(
            x=list(range(n_clusters)),
            y=[query_cluster_counts.get(i, 0) for i in range(n_clusters)],
            name="Query",
            marker_color="lightcoral",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Cluster ID",
        yaxis_title="Number of Tokens",
        barmode="group",
        width=width,
        height=height,
    )
    return fig


def visualize_multivector_compression(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    query_aggregated: dict[int, np.ndarray],
    doc_aggregated: dict[int, np.ndarray],
    query_tokens: list[str],
    doc_tokens: list[str],
    n_clusters: int = 8,
    width: int = 1000,
    height: int = 450,
) -> go.Figure:
    """
    Visualize MUVERA's dimensionality compression with side-by-side comparison.

    Shows the original multi-vector similarity matrix alongside the clustered
    representation, demonstrating how MUVERA compresses variable-length embeddings
    into fixed-dimensional encodings.

    Args:
        query_embeddings: Original query embeddings of shape (n_query_tokens, dim)
        doc_embeddings: Original document embeddings of shape (n_doc_tokens, dim)
        query_aggregated: Dictionary mapping cluster_id -> aggregated query vector
        doc_aggregated: Dictionary mapping cluster_id -> aggregated doc vector
        query_tokens: List of query token strings
        doc_tokens: List of document token strings
        n_clusters: Number of clusters (default 8)
        width: Figure width in pixels
        height: Figure height in pixels

    Returns:
        Plotly Figure object with side-by-side heatmap comparison

    Example:
        >>> fig = visualize_multivector_compression(
        ...     query_embeddings, doc_embeddings,
        ...     query_aggregated, doc_aggregated,
        ...     query_tokens, doc_tokens
        ... )
        >>> fig.show()
    """
    from plotly.subplots import make_subplots

    # Create side-by-side comparison
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Original Multi-Vector (Variable Length)",
            "MUVERA Clustered (Fixed Clusters)",
        ),
        horizontal_spacing=0.15,
    )

    # LEFT: Original multi-vector structure
    similarity_matrix = query_embeddings @ doc_embeddings.T
    fig.add_trace(
        go.Heatmap(
            z=similarity_matrix,
            x=doc_tokens,
            y=query_tokens,
            colorscale="YlGnBu",
            showscale=False,
            hovertemplate="Q: %{y}<br>D: %{x}<br>Sim: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # RIGHT: MUVERA clustered representation
    cluster_sims = np.zeros((n_clusters, n_clusters))
    for q_cluster in range(n_clusters):
        for d_cluster in range(n_clusters):
            cluster_sims[q_cluster, d_cluster] = np.dot(
                query_aggregated[q_cluster], doc_aggregated[d_cluster]
            )

    fig.add_trace(
        go.Heatmap(
            z=cluster_sims,
            x=[f"C{i}" for i in range(n_clusters)],
            y=[f"C{i}" for i in range(n_clusters)],
            colorscale="YlGnBu",
            showscale=True,
            hovertemplate="Q Cluster: %{y}<br>D Cluster: %{x}<br>Sim: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Document Tokens", row=1, col=1)
    fig.update_yaxes(title_text="Query Tokens", row=1, col=1)
    fig.update_xaxes(title_text="Document Clusters", row=1, col=2)
    fig.update_yaxes(title_text="Query Clusters", row=1, col=2)

    fig.update_layout(
        width=width,
        height=height,
        title_text=f"Dimensionality Compression: {len(query_tokens)}×{len(doc_tokens)} → {n_clusters}×{n_clusters}",
    )
    return fig


def hamming_distance(a: int, b: int) -> int:
    """
    Calculate Hamming distance between two cluster IDs.

    The Hamming distance measures how many bits differ between two integers
    when represented in binary. Used in MUVERA for finding nearest clusters
    when filling empty document clusters.

    Args:
        a: First cluster ID (integer)
        b: Second cluster ID (integer)

    Returns:
        Number of differing bits between the two cluster IDs

    Example:
        >>> hamming_distance(5, 7)  # Binary: 101 vs 111
        1
        >>> hamming_distance(0, 7)  # Binary: 000 vs 111
        3
    """
    xor = a ^ b
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1
    return count


def aggregate_document_clusters(
    embeddings: np.ndarray,
    cluster_ids: np.ndarray,
    n_clusters: int = 8,
) -> dict[int, np.ndarray]:
    """
    Aggregate document token embeddings by cluster using AVERAGE.

    MUVERA documents use averaging to create stable cluster representations.
    Empty clusters are filled using the vector from the nearest non-empty
    cluster (measured by Hamming distance).

    Args:
        embeddings: Token embeddings of shape (n_tokens, embedding_dim)
        cluster_ids: Cluster assignment for each token
        n_clusters: Total number of clusters (default 8)

    Returns:
        Dictionary mapping cluster_id -> aggregated vector (averaged)

    Example:
        >>> doc_embeddings = model.embed([doc_text], is_doc=True)[0]
        >>> cluster_ids = simhash.get_cluster_ids(doc_embeddings)
        >>> aggregated = aggregate_document_clusters(doc_embeddings, cluster_ids)
    """
    aggregated = {}

    # First pass: average embeddings for non-empty clusters
    for cluster_id in range(n_clusters):
        mask = cluster_ids == cluster_id
        if mask.any():
            cluster_embeddings = embeddings[mask]
            aggregated[cluster_id] = cluster_embeddings.mean(axis=0)
        else:
            aggregated[cluster_id] = None

    # Second pass: fill empty clusters with nearest non-empty cluster
    for cluster_id in range(n_clusters):
        if aggregated[cluster_id] is None:
            min_distance = float("inf")
            nearest_cluster = None
            for other_id in range(n_clusters):
                if aggregated[other_id] is not None:
                    dist = hamming_distance(cluster_id, other_id)
                    if dist < min_distance:
                        min_distance = dist
                        nearest_cluster = other_id
            if nearest_cluster is not None:
                aggregated[cluster_id] = aggregated[nearest_cluster].copy()
            else:
                # Fallback if somehow all clusters are empty
                aggregated[cluster_id] = np.zeros(embeddings.shape[1])

    return aggregated


def aggregate_query_clusters(
    embeddings: np.ndarray,
    cluster_ids: np.ndarray,
    n_clusters: int = 8,
) -> dict[int, np.ndarray]:
    """
    Aggregate query token embeddings by cluster using SUM.

    MUVERA queries use summing to preserve term frequency information.
    Empty clusters remain as zero vectors since queries are shorter and
    filling could introduce noise.

    Args:
        embeddings: Token embeddings of shape (n_tokens, embedding_dim)
        cluster_ids: Cluster assignment for each token
        n_clusters: Total number of clusters (default 8)

    Returns:
        Dictionary mapping cluster_id -> aggregated vector (summed)

    Example:
        >>> query_embeddings = model.embed([query_text], is_doc=False)[0]
        >>> cluster_ids = simhash.get_cluster_ids(query_embeddings)
        >>> aggregated = aggregate_query_clusters(query_embeddings, cluster_ids)
    """
    aggregated = {}

    for cluster_id in range(n_clusters):
        mask = cluster_ids == cluster_id
        if mask.any():
            cluster_embeddings = embeddings[mask]
            aggregated[cluster_id] = cluster_embeddings.sum(axis=0)
        else:
            # Empty cluster: leave as zero vector for queries
            aggregated[cluster_id] = np.zeros(embeddings.shape[1])

    return aggregated


def visualize_muvera_token_clustering(
    tokens: list[str],
    cluster_ids: np.ndarray,
    n_clusters: int = 8,
    title: str = "Token Clustering",
) -> HTML:
    """
    Visualize how MUVERA assigns tokens to clusters in an IPython table.

    Creates an HTML table where each cluster is represented as a column
    containing all tokens assigned to it. Empty clusters are visually distinct.

    Args:
        tokens: List of token strings
        cluster_ids: Array of cluster assignments for each token
        n_clusters: Total number of clusters (default 8)
        title: Table title

    Returns:
        IPython HTML display object showing tokens grouped by cluster

    Example:
        >>> doc_tokens = tokenize_late_interaction(model, doc_text, is_doc=True)
        >>> doc_embeddings = list(model.embed([doc_text], is_doc=True))[0]
        >>> cluster_ids = simhash.get_cluster_ids(doc_embeddings)
        >>> table = visualize_muvera_token_clustering(doc_tokens, cluster_ids)
        >>> display(table)
    """
    # Group tokens by cluster
    cluster_tokens = {}
    for cluster_id in range(n_clusters):
        mask = cluster_ids == cluster_id
        cluster_tokens[cluster_id] = [tokens[i] for i, m in enumerate(mask) if m]

    # Find max tokens in any cluster for row count
    max_tokens = (
        max(len(toks) for toks in cluster_tokens.values()) if cluster_tokens else 0
    )

    # Build table HTML
    html = f"""
    <div style="font-family: Arial, sans-serif; margin: 20px 0;">
        <h3 style="text-align: center; margin-bottom: 15px;">{title}</h3>
        <table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd;">
            <thead>
                <tr style="background-color: #1976D2; color: white;">
    """

    # Add headers
    for i in range(n_clusters):
        html += f'<th style="padding: 12px; text-align: center; border: 1px solid #ddd; font-weight: bold;">Cluster {i}</th>'

    html += """
                </tr>
            </thead>
            <tbody>
    """

    # Add rows
    for row_idx in range(max_tokens):
        html += "<tr>"
        for cluster_id in range(n_clusters):
            toks = cluster_tokens[cluster_id]

            # Determine background color
            if len(toks) == 0:
                bg_color = "#E0E0E0"  # Light gray for empty clusters
            else:
                bg_color = "#E3F2FD"  # Light blue for non-empty clusters

            # Get token or empty cell
            if row_idx < len(toks):
                token_text = toks[row_idx]
            else:
                token_text = ""

            html += f"<td style=\"padding: 8px; text-align: center; border: 1px solid #ddd; background-color: {bg_color}; font-family: 'Courier New', monospace; font-size: 12px;\">{token_text}</td>"

        html += "</tr>"

    html += """
            </tbody>
        </table>
    </div>
    """

    return HTML(html)


def visualize_muvera_aggregation(
    tokens: list[str],
    cluster_ids: np.ndarray,
    embeddings: np.ndarray,
    aggregated: dict[int, np.ndarray],
    is_document: bool = True,
    n_clusters: int = 8,
    title: str = "MUVERA Aggregation",
    width: int = 1400,
    height: int = 500,
) -> go.Figure:
    """
    Visualize MUVERA aggregation process with operation indicators.

    Creates a detailed table showing:
    - Cluster ID
    - Number of tokens per cluster
    - Aggregation operation (AVG for documents, SUM for queries)
    - Result magnitude after aggregation
    - Empty cluster handling (filled from nearest for docs, zero for queries)

    Args:
        tokens: List of token strings
        cluster_ids: Array of cluster assignments for each token
        embeddings: Token embeddings array
        aggregated: Dictionary mapping cluster_id -> aggregated vector
        is_document: True for document (AVG), False for query (SUM)
        n_clusters: Total number of clusters (default 8)
        title: Plot title
        width: Figure width in pixels
        height: Figure height in pixels

    Returns:
        Plotly Figure object with aggregation visualization

    Example:
        >>> doc_aggregated = aggregate_document_clusters(doc_embeddings, doc_cluster_ids)
        >>> fig = visualize_muvera_aggregation(
        ...     doc_tokens, doc_cluster_ids, doc_embeddings, doc_aggregated, is_document=True
        ... )
        >>> fig.show()
    """
    from collections import Counter

    # Count tokens per cluster
    cluster_counts = Counter(cluster_ids)

    # Prepare data for table
    cluster_labels = [f"C{i}" for i in range(n_clusters)]
    token_counts = [cluster_counts.get(i, 0) for i in range(n_clusters)]
    operations = ["AVG" if is_document else "SUM"] * n_clusters
    magnitudes = [f"{np.linalg.norm(aggregated[i]):.3f}" for i in range(n_clusters)]

    # Determine filling/empty status
    status = []
    cell_colors = []

    for cluster_id in range(n_clusters):
        count = cluster_counts.get(cluster_id, 0)

        if count == 0:
            if is_document:
                # Find which cluster this was filled from
                min_distance = float("inf")
                filled_from = None
                for other_id in range(n_clusters):
                    if cluster_counts.get(other_id, 0) > 0:
                        dist = hamming_distance(cluster_id, other_id)
                        if dist < min_distance:
                            min_distance = dist
                            filled_from = other_id
                status.append(
                    f"Filled from C{filled_from}"
                    if filled_from is not None
                    else "Empty"
                )
                cell_colors.append("#FFE082")  # Light orange for filled
            else:
                status.append("Empty (zero)")
                cell_colors.append("#E0E0E0")  # Gray for empty query
        else:
            status.append("Active")
            cell_colors.append("#C8E6C9")  # Light green for active

    # Create table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "<b>Cluster</b>",
                        "<b>Token Count</b>",
                        "<b>Operation</b>",
                        "<b>Magnitude</b>",
                        "<b>Status</b>",
                    ],
                    fill_color="#1976D2",
                    font=dict(color="white", size=14, family="Arial"),
                    align="center",
                    height=40,
                ),
                cells=dict(
                    values=[
                        cluster_labels,
                        token_counts,
                        operations,
                        magnitudes,
                        status,
                    ],
                    fill_color=[
                        ["#BBDEFB"] * n_clusters,  # Light blue for cluster column
                        ["#BBDEFB"] * n_clusters,  # Light blue for count column
                        ["#FFF9C4"] * n_clusters,  # Light yellow for operation column
                        ["#BBDEFB"] * n_clusters,  # Light blue for magnitude column
                        cell_colors,  # Dynamic colors for status column
                    ],
                    font=dict(color="black", size=13, family="Arial"),
                    align=[
                        "center",
                        "center",
                        "center",
                        "center",
                        "left",
                    ],
                    height=35,
                ),
            )
        ]
    )

    # Add subtitle explaining the operation
    operation_text = (
        "Documents: AVERAGE (stable representation) + Fill empty clusters"
        if is_document
        else "Queries: SUM (preserve term frequency) + Empty clusters remain zero"
    )

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>{operation_text}</sub>",
            x=0.5,
            xanchor="center",
        ),
        width=width,
        height=height,
    )

    return fig


def visualize_muvera_pipeline(
    doc_tokens: list[str],
    doc_cluster_ids: np.ndarray,
    doc_embeddings: np.ndarray,
    doc_aggregated: dict[int, np.ndarray],
    query_tokens: list[str],
    query_cluster_ids: np.ndarray,
    query_embeddings: np.ndarray,
    query_aggregated: dict[int, np.ndarray],
    n_clusters: int = 8,
) -> HTML:
    """
    Visualize complete MUVERA pipeline with document and query processing tables.

    Creates an HTML visualization showing:
    - Top: Document processing with AVG aggregation and empty cluster filling
    - Bottom: Query processing with SUM aggregation and empty clusters as zeros
    - Side-by-side comparison highlighting the key differences

    Args:
        doc_tokens: List of document token strings
        doc_cluster_ids: Document cluster assignments
        doc_embeddings: Document token embeddings
        doc_aggregated: Document aggregated vectors per cluster
        query_tokens: List of query token strings
        query_cluster_ids: Query cluster assignments
        query_embeddings: Query token embeddings
        query_aggregated: Query aggregated vectors per cluster
        n_clusters: Total number of clusters (default 8)

    Returns:
        IPython HTML display object with complete pipeline visualization

    Example:
        >>> doc_aggregated = aggregate_document_clusters(doc_embeddings, doc_cluster_ids)
        >>> query_aggregated = aggregate_query_clusters(query_embeddings, query_cluster_ids)
        >>> table = visualize_muvera_pipeline(
        ...     doc_tokens, doc_cluster_ids, doc_embeddings, doc_aggregated,
        ...     query_tokens, query_cluster_ids, query_embeddings, query_aggregated
        ... )
        >>> display(table)
    """
    from collections import Counter

    # Count tokens per cluster for both
    doc_cluster_counts = Counter(doc_cluster_ids)
    query_cluster_counts = Counter(query_cluster_ids)

    # Prepare data for both tables
    cluster_labels = [f"C{i}" for i in range(n_clusters)]

    # Document data
    doc_token_counts = [doc_cluster_counts.get(i, 0) for i in range(n_clusters)]
    doc_operations = ["AVG"] * n_clusters
    doc_magnitudes = [
        f"{np.linalg.norm(doc_aggregated[i]):.3f}" for i in range(n_clusters)
    ]
    doc_status = []
    doc_colors = []

    for cluster_id in range(n_clusters):
        count = doc_cluster_counts.get(cluster_id, 0)
        if count == 0:
            min_distance = float("inf")
            filled_from = None
            for other_id in range(n_clusters):
                if doc_cluster_counts.get(other_id, 0) > 0:
                    dist = hamming_distance(cluster_id, other_id)
                    if dist < min_distance:
                        min_distance = dist
                        filled_from = other_id
            doc_status.append(
                f"← Filled from C{filled_from}" if filled_from is not None else "Empty"
            )
            doc_colors.append("#FFE082")
        else:
            doc_status.append("Active")
            doc_colors.append("#C8E6C9")

    # Query data
    query_token_counts = [query_cluster_counts.get(i, 0) for i in range(n_clusters)]
    query_operations = ["SUM"] * n_clusters
    query_magnitudes = [
        f"{np.linalg.norm(query_aggregated[i]):.3f}" for i in range(n_clusters)
    ]
    query_status = []
    query_colors = []

    for cluster_id in range(n_clusters):
        count = query_cluster_counts.get(cluster_id, 0)
        if count == 0:
            query_status.append("Empty (zero)")
            query_colors.append("#E0E0E0")
        else:
            query_status.append("Active")
            query_colors.append("#C8E6C9")

    # Build HTML
    html = """
    <div style="font-family: Arial, sans-serif; margin: 20px 0;">
        <h3 style="text-align: center; margin-bottom: 20px;"><b>MUVERA Processing Pipeline: Document vs Query</b></h3>

        <!-- Document Table -->
        <h4 style="color: #1565C0; margin-top: 20px; margin-bottom: 10px;"><b>Document Processing: AVERAGE + Fill Empty Clusters</b></h4>
        <table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd; margin-bottom: 30px;">
            <thead>
                <tr style="background-color: #1565C0; color: white;">
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Cluster</th>
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Tokens</th>
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Op</th>
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Magnitude</th>
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Status</th>
                </tr>
            </thead>
            <tbody>
    """

    # Add document table rows
    for i in range(n_clusters):
        html += f"""
                <tr>
                    <td style="padding: 10px; text-align: center; border: 1px solid #ddd; background-color: #BBDEFB;">{cluster_labels[i]}</td>
                    <td style="padding: 10px; text-align: center; border: 1px solid #ddd; background-color: #BBDEFB;">{doc_token_counts[i]}</td>
                    <td style="padding: 10px; text-align: center; border: 1px solid #ddd; background-color: #FFF9C4;">{doc_operations[i]}</td>
                    <td style="padding: 10px; text-align: center; border: 1px solid #ddd; background-color: #BBDEFB;">{doc_magnitudes[i]}</td>
                    <td style="padding: 10px; text-align: left; border: 1px solid #ddd; background-color: {doc_colors[i]};">{doc_status[i]}</td>
                </tr>
        """

    html += """
            </tbody>
        </table>

        <!-- Query Table -->
        <h4 style="color: #C62828; margin-top: 20px; margin-bottom: 10px;"><b>Query Processing: SUM + Leave Empty as Zero</b></h4>
        <table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd;">
            <thead>
                <tr style="background-color: #C62828; color: white;">
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Cluster</th>
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Tokens</th>
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Op</th>
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Magnitude</th>
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Status</th>
                </tr>
            </thead>
            <tbody>
    """

    # Add query table rows
    for i in range(n_clusters):
        html += f"""
                <tr>
                    <td style="padding: 10px; text-align: center; border: 1px solid #ddd; background-color: #FFCDD2;">{cluster_labels[i]}</td>
                    <td style="padding: 10px; text-align: center; border: 1px solid #ddd; background-color: #FFCDD2;">{query_token_counts[i]}</td>
                    <td style="padding: 10px; text-align: center; border: 1px solid #ddd; background-color: #FFF9C4;">{query_operations[i]}</td>
                    <td style="padding: 10px; text-align: center; border: 1px solid #ddd; background-color: #FFCDD2;">{query_magnitudes[i]}</td>
                    <td style="padding: 10px; text-align: left; border: 1px solid #ddd; background-color: {query_colors[i]};">{query_status[i]}</td>
                </tr>
        """

    html += """
            </tbody>
        </table>
    </div>
    """

    return HTML(html)


def load_or_compute_query_embeddings(
    load_precomputed: bool = True,
    parquet_path: str = "ro_shared_data/pdfs/query-embeddings.parquet",
):
    """
    Load or compute query embeddings for ColPali.

    Either loads pre-computed query embeddings from a parquet file or computes
    them using the ColPali model. Returns a pandas DataFrame with query text
    and embeddings. The queries, model, and processor are defined internally.

    Args:
        load_precomputed: If True, load from parquet_path; if False, compute embeddings
        parquet_path: Path to parquet file with precomputed embeddings

    Returns:
        pandas DataFrame with columns: 'query' and 'query_embedding'

    Raises:
        FileNotFoundError: If load_precomputed=True but parquet file doesn't exist

    Example:
        >>> # Load precomputed embeddings
        >>> queries_df = load_or_compute_query_embeddings(
        ...     load_precomputed=True
        ... )
        >>> # Compute embeddings
        >>> queries_df = load_or_compute_query_embeddings(
        ...     load_precomputed=False
        ... )
    """
    import pandas as pd
    import torch

    # Define queries internally
    queries = [
        "coffee mug",
        "size vs performance tradeoff",
        "one learning algorithm",
    ]

    if not load_precomputed:
        from colpali_engine.models import ColPali, ColPaliProcessor

        # Load model and processor internally
        model_name = "vidore/colpali-v1.3"
        processor = ColPaliProcessor.from_pretrained(model_name)
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Process queries with the processor
        batch_queries = processor.process_queries(queries)

        # Generate embeddings using the model
        with torch.no_grad():
            query_embeddings = model(**batch_queries)

        # Convert to list format (double precision, CPU, numpy, then list)
        query_embeddings = [
            qe.double().cpu().numpy().tolist() for qe in query_embeddings
        ]

        # Create DataFrame
        queries_df = pd.DataFrame.from_dict(
            {
                "query": queries,
                "query_embedding": query_embeddings,
            }
        )

        # Save queries to the target path
        queries_df.to_parquet(parquet_path, engine="pyarrow")

    else:
        # Load precomputed embeddings
        try:
            queries_df = pd.read_parquet(parquet_path, engine="pyarrow")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Precomputed embeddings file not found at: {parquet_path}"
            )

    return queries_df


def load_or_compute_rag_query_embeddings(
    load_precomputed: bool = True,
    parquet_path: str = "ro_shared_data/pdfs/rag-query-embeddings.parquet",
):
    """
    Load or compute RAG query embeddings for ColPali (Lesson 5).

    Either loads pre-computed query embeddings from a parquet file or computes
    them using the ColPali model. Returns a pandas DataFrame with query text
    and embeddings. The queries are specific to the RAG lesson (L5).

    Args:
        load_precomputed: If True, load from parquet_path; if False, compute embeddings
        parquet_path: Path to parquet file with precomputed embeddings

    Returns:
        pandas DataFrame with columns: 'query' and 'query_embedding'

    Raises:
        FileNotFoundError: If load_precomputed=True but parquet file doesn't exist

    Example:
        >>> # Load precomputed embeddings
        >>> queries_df = load_or_compute_rag_query_embeddings(
        ...     load_precomputed=True
        ... )
        >>> # Compute embeddings
        >>> queries_df = load_or_compute_rag_query_embeddings(
        ...     load_precomputed=False
        ... )
    """
    import pandas as pd
    import torch

    # Define RAG queries internally (from L5)
    queries = [
        "Describe the concept of the 'one learning algorithm'",
        "Explain the size vs performance tradeoff",
        "What was the coffee mug example used to present?",
        "How does the human brain work?",
    ]

    if not load_precomputed:
        from colpali_engine.models import ColPali, ColPaliProcessor

        # Load model and processor internally
        model_name = "vidore/colpali-v1.3"
        processor = ColPaliProcessor.from_pretrained(model_name)
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Process queries with the processor
        batch_queries = processor.process_queries(queries)

        # Generate embeddings using the model
        with torch.no_grad():
            query_embeddings = model(**batch_queries)

        # Convert to list format (double precision, CPU, numpy, then list)
        query_embeddings = [
            qe.double().cpu().numpy().tolist() for qe in query_embeddings
        ]

        # Create DataFrame
        queries_df = pd.DataFrame.from_dict(
            {
                "query": queries,
                "query_embedding": query_embeddings,
            }
        )

        # Save queries to the target path
        queries_df.to_parquet(parquet_path, engine="pyarrow")

    else:
        # Load precomputed embeddings
        try:
            queries_df = pd.read_parquet(parquet_path, engine="pyarrow")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Precomputed embeddings file not found at: {parquet_path}"
            )

    return queries_df


def load_or_compute_image_embeddings(
    load_precomputed: bool = True,
    parquet_path: str = "ro_shared_data/pdfs/colpali-embeddings.parquet",
):
    """
    Load or compute image embeddings for ColPali from PDF documents.

    Either loads pre-computed image embeddings from parquet files or computes
    them by converting PDFs to screenshots and generating embeddings using ColPali.
    Returns a pandas DataFrame with image paths and embeddings as numpy arrays.
    The PDF conversion, model loading, and embedding generation are all handled
    internally.

    When computing embeddings, the function checks if PNG screenshots already exist
    in ro_shared_data/pdfs/screenshots/. If found, PDF conversion is skipped to save
    time. Embeddings are saved in chunks of 100 images each to minimize memory usage.
    Chunks are saved as separate parquet files with names like:
    colpali-embeddings-chunk-001.parquet, colpali-embeddings-chunk-002.parquet, etc.

    When loading, the function automatically detects and concatenates all chunk files.

    Args:
        load_precomputed: If True, load from chunk files; if False, compute embeddings
        parquet_path: Base path for parquet files (used to derive chunk file names)

    Returns:
        pandas DataFrame with columns: 'image_path' and 'image_embedding' (as numpy arrays)

    Raises:
        FileNotFoundError: If load_precomputed=True but no chunk files or single file exist

    Example:
        >>> # Load precomputed embeddings from chunks
        >>> images_df = load_or_compute_image_embeddings(
        ...     load_precomputed=True
        ... )
        >>> # Compute embeddings and save in chunks
        >>> images_df = load_or_compute_image_embeddings(
        ...     load_precomputed=False
        ... )
    """
    import pandas as pd
    import numpy as np

    if not load_precomputed:
        import torch
        from pathlib import Path
        from PIL import Image
        from tqdm import tqdm
        from colpali_engine.models import ColPali, ColPaliProcessor

        # Convert PDFs to screenshots (if not already done)
        input_dir = Path("ro_shared_data/pdfs")
        output_dir = Path("ro_shared_data/pdfs/screenshots")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if PNG screenshots already exist
        existing_pngs = list(output_dir.glob("*.png"))
        if existing_pngs:
            print(
                f"Found {len(existing_pngs)} existing PNG screenshots, skipping PDF conversion."
            )
        else:
            print("No existing PNG screenshots found, converting PDFs...")
            for file_path in tqdm(
                list(input_dir.glob("*.pdf")), desc="Converting PDFs"
            ):
                page_iterator = pdf_to_png_screenshots(file_path)
                for i, page_image in enumerate(page_iterator, start=1):
                    output_file = output_dir / f"{file_path.stem}-page-{i:05d}.png"
                    page_image.save(output_file)

        # Load ColPali model and processor
        model_name = "vidore/colpali-v1.3"
        processor = ColPaliProcessor.from_pretrained(model_name)
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Get image paths (images will be loaded lazily in batches)
        image_paths = sorted(output_dir.glob("*.png"))

        # Generate embeddings in batches and save in chunks
        import gc

        batch_size = 4
        chunk_size = 100  # Save every 100 images
        chunk_image_paths = []
        chunk_embeddings = []
        chunk_num = 1
        processed_count = 0

        # Prepare chunk file pattern
        base_path = Path(parquet_path)
        chunk_dir = base_path.parent
        chunk_dir.mkdir(parents=True, exist_ok=True)

        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        for batch_paths in tqdm(
            batched(image_paths, batch_size),
            desc="Generating embeddings",
            total=num_batches,
        ):
            # Load images just-in-time for this batch
            batch_images_list = [Image.open(path) for path in batch_paths]

            # Process batch
            batch_images = processor.process_images(batch_images_list).to(model.device)
            with torch.no_grad():
                batch_embeddings = model(**batch_images)

                # Accumulate embeddings for current chunk
                chunk_embeddings += batch_embeddings.tolist()
                chunk_image_paths.extend(batch_paths)
                processed_count += len(batch_paths)

            # Clean up to free memory
            for img in batch_images_list:
                img.close()
            del batch_images_list, batch_images, batch_embeddings
            gc.collect()

            # Save chunk when we reach chunk_size images
            if len(chunk_image_paths) >= chunk_size:
                chunk_df = pd.DataFrame.from_dict(
                    {
                        "image_path": list(map(str, chunk_image_paths)),
                        "image_embedding": chunk_embeddings,
                    }
                )
                chunk_file = (
                    chunk_dir / f"{base_path.stem}-chunk-{chunk_num:03d}.parquet"
                )
                chunk_df.to_parquet(chunk_file, engine="pyarrow")

                # Reset chunk accumulators
                chunk_image_paths = []
                chunk_embeddings = []
                chunk_num += 1
                gc.collect()

        # Save any remaining images in final chunk
        if chunk_image_paths:
            chunk_df = pd.DataFrame.from_dict(
                {
                    "image_path": list(map(str, chunk_image_paths)),
                    "image_embedding": chunk_embeddings,
                }
            )
            chunk_file = chunk_dir / f"{base_path.stem}-chunk-{chunk_num:03d}.parquet"
            chunk_df.to_parquet(chunk_file, engine="pyarrow")
            gc.collect()

        # Load the chunks we just saved into images_df
        chunk_pattern = f"{base_path.stem}-chunk-*.parquet"
        chunk_files = sorted(chunk_dir.glob(chunk_pattern))
        chunk_dfs = []
        for chunk_file in chunk_files:
            chunk_dfs.append(pd.read_parquet(chunk_file, engine="pyarrow"))
        images_df = pd.concat(chunk_dfs, ignore_index=True)

    else:
        # Load precomputed embeddings from chunk files
        from pathlib import Path

        base_path = Path(parquet_path)
        chunk_dir = base_path.parent
        chunk_pattern = f"{base_path.stem}-chunk-*.parquet"

        # Find all chunk files
        chunk_files = sorted(chunk_dir.glob(chunk_pattern))

        if chunk_files:
            # Load and concatenate all chunks
            chunk_dfs = []
            for chunk_file in chunk_files:
                chunk_dfs.append(pd.read_parquet(chunk_file, engine="pyarrow"))
            images_df = pd.concat(chunk_dfs, ignore_index=True)
        else:
            # Fall back to single file for backward compatibility
            try:
                images_df = pd.read_parquet(parquet_path, engine="pyarrow")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Precomputed embeddings not found. Expected either:\n"
                    f"  - Chunk files: {chunk_dir / chunk_pattern}\n"
                    f"  - Single file: {parquet_path}"
                )

    # Convert embeddings to numpy arrays
    images_df["image_embedding"] = images_df["image_embedding"].apply(np.stack)

    return images_df


def load_sample_image_embeddings(
    load_precomputed: bool = True,
    parquet_path: str = "ro_shared_data/pdfs/colpali-embeddings.parquet",
):
    """
    Load or compute a SAMPLE of image embeddings (100 images) for ColPali from PDF documents.

    This is a memory-efficient version of load_or_compute_image_embeddings that loads
    only the first 100 images (~19MB instead of 330MB for all 1,737 images). Use this
    when you need to test or demonstrate functionality without loading the full dataset.

    Either loads pre-computed image embeddings from the first parquet chunk file or computes
    embeddings for the first 100 images by converting PDFs to screenshots and generating
    embeddings using ColPali. Returns a pandas DataFrame with image paths and embeddings
    as numpy arrays.

    When computing embeddings, the function checks if PNG screenshots already exist
    in ro_shared_data/pdfs/screenshots/. If found, PDF conversion is skipped to save
    time. Only the first 100 images are processed and saved as a single chunk.

    When loading, the function loads only the first chunk file (colpali-embeddings-chunk-001.parquet)
    instead of all 18 chunks.

    Args:
        load_precomputed: If True, load from first chunk file; if False, compute embeddings for first 100 images
        parquet_path: Base path for parquet files (used to derive chunk file names)

    Returns:
        pandas DataFrame with columns: 'image_path' and 'image_embedding' (as numpy arrays)
        Limited to 100 images maximum

    Raises:
        FileNotFoundError: If load_precomputed=True but first chunk file or single file don't exist

    Example:
        >>> # Load first 100 precomputed embeddings
        >>> images_df = load_sample_image_embeddings(
        ...     load_precomputed=True
        ... )
        >>> # Compute embeddings for first 100 images
        >>> images_df = load_sample_image_embeddings(
        ...     load_precomputed=False
        ... )
    """
    import pandas as pd
    import numpy as np

    SAMPLE_SIZE = 100  # Number of images to load/compute

    if not load_precomputed:
        import torch
        from pathlib import Path
        from PIL import Image
        from tqdm import tqdm
        from colpali_engine.models import ColPali, ColPaliProcessor

        # Convert PDFs to screenshots (if not already done)
        input_dir = Path("ro_shared_data/pdfs")
        output_dir = Path("ro_shared_data/pdfs/screenshots")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if PNG screenshots already exist
        existing_pngs = sorted(output_dir.glob("*.png"))[:SAMPLE_SIZE]
        if existing_pngs:
            print(
                f"Found existing PNG screenshots, will process first {len(existing_pngs)} images."
            )
            image_paths = existing_pngs
        else:
            print("No existing PNG screenshots found, converting first PDFs...")
            pdf_files = sorted(input_dir.glob("*.pdf"))
            image_paths = []
            images_generated = 0

            for file_path in tqdm(pdf_files, desc="Converting PDFs"):
                if images_generated >= SAMPLE_SIZE:
                    break
                page_iterator = pdf_to_png_screenshots(file_path)
                for i, page_image in enumerate(page_iterator, start=1):
                    if images_generated >= SAMPLE_SIZE:
                        break
                    output_file = output_dir / f"{file_path.stem}-page-{i:05d}.png"
                    page_image.save(output_file)
                    image_paths.append(output_file)
                    images_generated += 1

        # Limit to SAMPLE_SIZE images
        image_paths = image_paths[:SAMPLE_SIZE]

        # Load ColPali model and processor
        model_name = "vidore/colpali-v1.3"
        processor = ColPaliProcessor.from_pretrained(model_name)
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Generate embeddings in batches
        import gc

        batch_size = 4
        all_image_paths = []
        all_embeddings = []

        # Prepare output path
        base_path = Path(parquet_path)
        chunk_dir = base_path.parent
        chunk_dir.mkdir(parents=True, exist_ok=True)

        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        for batch_paths in tqdm(
            batched(image_paths, batch_size),
            desc="Generating embeddings (sample)",
            total=num_batches,
        ):
            # Load images just-in-time for this batch
            batch_images_list = [Image.open(path) for path in batch_paths]

            # Process batch
            batch_images = processor.process_images(batch_images_list).to(model.device)
            with torch.no_grad():
                batch_embeddings = model(**batch_images)

                # Accumulate embeddings
                all_embeddings += batch_embeddings.tolist()
                all_image_paths.extend(batch_paths)

            # Clean up to free memory
            for img in batch_images_list:
                img.close()
            del batch_images_list, batch_images, batch_embeddings
            gc.collect()

        # Save as single chunk
        chunk_df = pd.DataFrame.from_dict(
            {
                "image_path": list(map(str, all_image_paths)),
                "image_embedding": all_embeddings,
            }
        )
        chunk_file = chunk_dir / f"{base_path.stem}-sample-chunk-001.parquet"
        chunk_df.to_parquet(chunk_file, engine="pyarrow")

        images_df = chunk_df
        gc.collect()

    else:
        # Load precomputed embeddings from FIRST chunk file only
        from pathlib import Path

        base_path = Path(parquet_path)
        chunk_dir = base_path.parent

        # Try to load first chunk file
        first_chunk_file = chunk_dir / f"{base_path.stem}-chunk-001.parquet"

        if first_chunk_file.exists():
            # Load only the first chunk (100 images)
            images_df = pd.read_parquet(first_chunk_file, engine="pyarrow")
        else:
            # Fall back to loading first 100 rows from single file
            try:
                full_df = pd.read_parquet(parquet_path, engine="pyarrow")
                images_df = full_df.head(SAMPLE_SIZE)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Precomputed embeddings not found. Expected either:\n"
                    f"  - First chunk file: {first_chunk_file}\n"
                    f"  - Single file: {parquet_path}\n"
                    f"(Will load first {SAMPLE_SIZE} images only)"
                )

    # Convert embeddings to numpy arrays
    images_df["image_embedding"] = images_df["image_embedding"].apply(np.stack)

    return images_df


def yield_optimized_embeddings(
    load_precomputed: bool = True,
    embeddings_dir: str = "ro_shared_data/pdfs",
    output_dir: str = "ro_shared_data/pdfs/optimized",
    allowed_docs=None,
):
    """
    Generator that yields optimized ColPali embeddings for efficient Qdrant upsertion.

    This function provides two modes:
    1. Load mode (load_precomputed=True): Loads pre-computed optimized embeddings from
       parquet files and yields them one by one without creating DataFrames.
    2. Compute mode (load_precomputed=False): Loads original embeddings, computes all
       optimization variants, saves them to new parquet files, and yields them.

    The function processes all 18 parquet chunks (1,737 images total) and yields each
    image embedding with all optimization variants applied.

    Optimization variants computed:
    - original: Full ColPali embeddings (~1031 tokens × 128 dims)
    - hierarchical_2x: Hierarchical token pooling with factor 2 (~515 tokens)
    - hierarchical_4x: Hierarchical token pooling with factor 4 (~257 tokens)
    - row_pooled: Row-wise mean pooling (32 tokens, horizontal patterns)
    - column_pooled: Column-wise mean pooling (32 tokens, vertical patterns)

    Args:
        load_precomputed: If True, load from optimized parquet files; if False,
                         compute optimizations from original embeddings and save
        embeddings_dir: Directory containing original embedding parquet files
                       (expects colpali-embeddings-chunk-*.parquet files)
        output_dir: Directory to save optimized parquet files when computing
                   (creates colpali-embeddings-optimized-chunk-*.parquet files)
        allowed_docs: Optional tuple/list of document name patterns to filter by.
                     If provided, only yields embeddings for images whose paths
                     contain any of the specified patterns (substring matching).
                     If None, yields all embeddings. Example: ("MLS_C2_W1", "DLS_C1_W4")

    Yields:
        Tuple of (image_path, vectors_dict) where:
        - image_path: str, path to the image file
        - vectors_dict: dict with keys 'original', 'hierarchical_2x',
                       'hierarchical_4x', 'row_pooled', 'column_pooled'
                       and numpy array values

    Example:
        >>> # First time: compute and save optimizations
        >>> for image_path, vectors in yield_optimized_embeddings(load_precomputed=False):
        ...     client.upsert(collection, points=[{
        ...         "vector": vectors,
        ...         "payload": {"image_path": image_path}
        ...     }])

        >>> # Subsequent times: load precomputed optimizations
        >>> for image_path, vectors in yield_optimized_embeddings(load_precomputed=True):
        ...     # Fast loading, no computation needed
        ...     client.upsert(...)
    """
    from pathlib import Path
    import pandas as pd

    embeddings_path = Path(embeddings_dir)

    if load_precomputed:
        # Load mode: read from optimized parquet files
        output_path = Path(output_dir)
        optimized_files = sorted(
            output_path.glob("colpali-embeddings-optimized-chunk-*.parquet")
        )

        if not optimized_files:
            raise FileNotFoundError(
                f"No optimized embedding files found in {output_path}/\n"
                f"Expected files like: colpali-embeddings-optimized-chunk-001.parquet\n"
                f"Run with load_precomputed=False first to generate them."
            )

        # Yield from each chunk file
        for chunk_file in optimized_files:
            chunk_df = pd.read_parquet(chunk_file, engine="pyarrow")

            # Convert embeddings to numpy arrays if needed
            for col in ["original", "hierarchical_2x", "hierarchical_4x", "row_pooled", "column_pooled"]:
                if col in chunk_df.columns:
                    chunk_df[col] = chunk_df[col].apply(np.stack)

            # Yield each row
            for _, row in chunk_df.iterrows():
                # Filter by allowed documents if specified
                if allowed_docs is not None:
                    is_allowed = any(fname in row["image_path"] for fname in allowed_docs)
                    if not is_allowed:
                        continue

                vectors_dict = {
                    "original": row["original"],
                    "hierarchical_2x": row["hierarchical_2x"],
                    "hierarchical_4x": row["hierarchical_4x"],
                    "row_pooled": row["row_pooled"],
                    "column_pooled": row["column_pooled"],
                }
                yield row["image_path"], vectors_dict

    else:
        # Compute mode: load original embeddings, compute optimizations, save and yield
        import torch
        from colpali_engine.compression.token_pooling import HierarchicalTokenPooler
        from colpali_engine.models import ColPaliProcessor
        from tqdm import tqdm

        # Find all original embedding chunk files
        chunk_files = sorted(embeddings_path.glob("colpali-embeddings-chunk-*.parquet"))

        if not chunk_files:
            raise FileNotFoundError(
                f"No original embedding files found in {embeddings_path}/\n"
                f"Expected files like: colpali-embeddings-chunk-001.parquet"
            )

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize pooler for hierarchical token pooling
        pooler = HierarchicalTokenPooler()

        # Get image mask from processor (same for all images)
        model_name = "vidore/colpali-v1.3"
        processor = ColPaliProcessor.from_pretrained(model_name)

        # Helper functions for optimizations
        def hierarchical_token_pooling(arr: np.ndarray, pool_factor: int = 2) -> np.ndarray:
            arr_tensor = torch.from_numpy(arr[np.newaxis, :, :])
            pooled = pooler.pool_embeddings(arr_tensor, pool_factor=pool_factor)
            return pooled.cpu().detach().numpy()[0]

        def embeddings_grid(image_embeddings: np.ndarray):
            patch_size = 32
            model_dim = 128
            return image_embeddings.reshape((patch_size, patch_size, model_dim))

        def row_mean_pooling(grid_embeddings: np.ndarray) -> np.ndarray:
            return grid_embeddings.mean(axis=1)

        def column_mean_pooling(grid_embeddings: np.ndarray) -> np.ndarray:
            return grid_embeddings.mean(axis=0)

        # Process each chunk file
        for chunk_file in tqdm(chunk_files, desc="Processing chunks"):
            # Load original embeddings
            chunk_df = pd.read_parquet(chunk_file, engine="pyarrow")
            chunk_df["image_embedding"] = chunk_df["image_embedding"].apply(np.stack)

            # Get image mask using first image from this chunk
            first_image = Image.open(chunk_df["image_path"].iloc[0])
            batch_images = processor.process_images([first_image])
            image_mask = processor.get_image_mask(batch_images)[0]

            # Process each image in the chunk
            optimized_rows = []
            for _, row in tqdm(
                chunk_df.iterrows(),
                total=len(chunk_df),
                desc=f"Processing {chunk_file.name}",
                leave=False
            ):
                # Filter by allowed documents if specified
                if allowed_docs is not None:
                    is_allowed = any(fname in row["image_path"] for fname in allowed_docs)
                    if not is_allowed:
                        continue

                # Extract image-only embeddings and create grid
                grid = embeddings_grid(row["image_embedding"][image_mask])

                # Compute all optimizations
                vectors_dict = {
                    "original": row["image_embedding"],
                    "hierarchical_2x": hierarchical_token_pooling(row["image_embedding"], 2),
                    "hierarchical_4x": hierarchical_token_pooling(row["image_embedding"], 4),
                    "row_pooled": row_mean_pooling(grid),
                    "column_pooled": column_mean_pooling(grid),
                }

                # Store for saving to parquet (convert arrays to lists for PyArrow compatibility)
                optimized_rows.append({
                    "image_path": row["image_path"],
                    "original": vectors_dict["original"].tolist(),
                    "hierarchical_2x": vectors_dict["hierarchical_2x"].tolist(),
                    "hierarchical_4x": vectors_dict["hierarchical_4x"].tolist(),
                    "row_pooled": vectors_dict["row_pooled"].tolist(),
                    "column_pooled": vectors_dict["column_pooled"].tolist(),
                })

                # Yield immediately so caller can start upserting (keep as numpy arrays)
                yield row["image_path"], vectors_dict

            # Save optimized chunk to parquet
            optimized_df = pd.DataFrame(optimized_rows)
            chunk_number = chunk_file.stem.split("-")[-1]  # Extract "001" from "chunk-001"
            output_file = output_path / f"colpali-embeddings-optimized-chunk-{chunk_number}.parquet"
            optimized_df.to_parquet(output_file, engine="pyarrow")
            print(f"Saved {output_file}")


def yield_muvera_embeddings(
    muvera,
    load_precomputed: bool = True,
    embeddings_dir: str = "ro_shared_data/pdfs",
):
    """
    Generator that yields MUVERA-compressed ColPali embeddings for efficient Qdrant upsertion.

    This function streams ColPali embeddings chunk-by-chunk, computes MUVERA compression
    on-the-fly, and yields both original and MUVERA embeddings. Designed for memory-efficient
    processing of large datasets without loading all embeddings into memory.

    The function processes all 18 parquet chunks (1,737 images total) and yields each
    image with both its original ColPali multi-vector embeddings and the computed
    MUVERA fixed-dimensional encoding (FDE).

    Args:
        muvera: Muvera instance from fastembed.postprocess.muvera, pre-configured
               with desired parameters (k_sim, dim_proj, r_reps, etc.)
        load_precomputed: If True, load from precomputed ColPali embedding files;
                         if False, compute ColPali embeddings from scratch
                         (Note: MUVERA is always computed on-the-fly)
        embeddings_dir: Directory containing ColPali embedding parquet files
                       (expects colpali-embeddings-chunk-*.parquet files)

    Yields:
        Tuple of (image_path, vectors_dict) where:
        - image_path: str, path to the image file
        - vectors_dict: dict with keys:
            - "colpali_original": numpy array of shape (n_tokens, 128), original multi-vector embeddings
            - "muvera_fde": numpy array of shape (20480,), MUVERA fixed-dimensional encoding

    Example:
        >>> from fastembed.postprocess.muvera import Muvera
        >>> muvera = Muvera(dim=128, k_sim=6, dim_proj=16, r_reps=20, random_seed=42)
        >>>
        >>> for i, (image_path, vectors) in enumerate(yield_muvera_embeddings(muvera)):
        ...     client.upsert(collection, points=[
        ...         models.PointStruct(
        ...             id=i,
        ...             vector={
        ...                 "colpali_original": vectors["colpali_original"],
        ...                 "muvera_fde": vectors["muvera_fde"],
        ...             },
        ...             payload={"image_path": image_path}
        ...         )
        ...     ])
    """
    from pathlib import Path
    import pandas as pd

    embeddings_path = Path(embeddings_dir)

    if load_precomputed:
        # Load mode: read from precomputed ColPali embeddings and compute MUVERA on-the-fly
        chunk_files = sorted(embeddings_path.glob("colpali-embeddings-chunk-*.parquet"))

        if not chunk_files:
            raise FileNotFoundError(
                f"No ColPali embedding files found in {embeddings_path}/\n"
                f"Expected files like: colpali-embeddings-chunk-001.parquet\n"
                f"Run with load_precomputed=False to generate them."
            )

        # Yield from each chunk file
        for chunk_file in chunk_files:
            chunk_df = pd.read_parquet(chunk_file, engine="pyarrow")

            # Convert embeddings to numpy arrays
            chunk_df["image_embedding"] = chunk_df["image_embedding"].apply(np.stack)

            # Process each image in the chunk
            for _, row in chunk_df.iterrows():
                colpali_original = row["image_embedding"]

                # Compute MUVERA FDE on-the-fly
                muvera_fde = muvera.process_document(colpali_original)

                vectors_dict = {
                    "colpali_original": colpali_original,
                    "muvera_fde": muvera_fde,
                }

                yield row["image_path"], vectors_dict

    else:
        # Compute mode: generate ColPali embeddings from scratch, then compute MUVERA
        # This would require loading images and running the ColPali model
        # For now, delegate to existing load_sample_image_embeddings logic
        raise NotImplementedError(
            "Computing ColPali embeddings from scratch is not yet implemented in yield_muvera_embeddings.\n"
            "Please use load_precomputed=True to work with existing embeddings, or use the notebook's\n"
            "existing ColPali embedding generation code."
        )


def load_or_compute_attention_embeddings(
    load_precomputed: bool = True,
    model_name: str = None,
    parquet_path: str = None,
):
    """
    Load or compute embeddings for the "Attention is All You Need" paper images.

    Either loads pre-computed embeddings from a parquet file or computes them
    using ColPali/ColSMOL models. This function is designed for the 10-page
    attention-is-all-you-need paper in ro_shared_data. Returns a pandas DataFrame
    with file paths and embeddings as numpy arrays.

    The function supports two models:
    - vidore/colSmol-256M (CPU-friendly, default)
    - vidore/colpali-v1.3 (GPU, more powerful)

    Cache files are automatically named based on the model:
    - attention-embeddings-colsmol.parquet (for colSmol)
    - attention-embeddings-colpali.parquet (for colpali)

    Args:
        load_precomputed: If True, load from parquet_path; if False, compute embeddings
        model_name: Model name to use (auto-detects CUDA if None)
        parquet_path: Path to parquet file (auto-determined from model_name if None)

    Returns:
        pandas DataFrame with columns: 'file_path' and 'image_embedding' (as numpy arrays)

    Raises:
        FileNotFoundError: If load_precomputed=True but parquet file doesn't exist

    Example:
        >>> # Load precomputed embeddings (auto-detects model)
        >>> df = load_or_compute_attention_embeddings(load_precomputed=True)
        >>> # Compute embeddings for specific model
        >>> df = load_or_compute_attention_embeddings(
        ...     load_precomputed=False,
        ...     model_name="vidore/colpali-v1.3"
        ... )
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path

    # Auto-detect model name if not provided
    if model_name is None:
        # Auto-detect based on CUDA availability
        import torch
        if torch.cuda.is_available():
            model_name = "vidore/colpali-v1.3"
        else:
            model_name = "vidore/colSmol-256M"

    # Auto-determine cache file path based on model
    if parquet_path is None:
        if "colSmol" in model_name or "colsmol" in model_name.lower():
            cache_name = "attention-embeddings-colsmol.parquet"
        elif "colpali" in model_name.lower():
            cache_name = "attention-embeddings-colpali.parquet"
        else:
            cache_name = "attention-embeddings.parquet"
        parquet_path = f"ro_shared_data/attention-is-all-you-need/{cache_name}"

    if not load_precomputed:
        import torch
        from glob import glob
        from tqdm import tqdm
        from PIL import Image

        # Load model and processor
        if "colSmol" in model_name or "colsmol" in model_name.lower():
            from colpali_engine.models import ColIdefics3Processor, ColIdefics3
            processor = ColIdefics3Processor.from_pretrained(model_name)
            model = ColIdefics3.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                device_map="cpu",
            )
        else:
            from colpali_engine.models import ColPaliProcessor, ColPali
            processor = ColPaliProcessor.from_pretrained(model_name)
            model = ColPali.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )

        # Get all PNG files from attention paper directory
        image_paths = sorted(glob("ro_shared_data/attention-is-all-you-need/*.png"))

        file_paths = []
        embeddings = []

        # Generate embeddings for each image
        for file_path in tqdm(image_paths, desc="Generating attention paper embeddings"):
            # Load image
            image = Image.open(file_path)

            # Process through ColPali/ColSMOL
            batch_images = processor.process_images([image]).to(model.device)
            with torch.no_grad():
                image_embeddings = model(**batch_images).to(dtype=torch.float32)

            # Extract image-only embeddings using mask
            image_mask = processor.get_image_mask(batch_images)
            masked_image_embeddings = image_embeddings[image_mask]

            # Store results (convert to list for parquet compatibility)
            file_paths.append(file_path)
            embeddings.append(masked_image_embeddings.cpu().numpy().tolist())

            # Clean up
            image.close()

        # Create DataFrame
        embeddings_df = pd.DataFrame({
            "file_path": file_paths,
            "image_embedding": embeddings,
        })

        # Save to parquet
        Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
        embeddings_df.to_parquet(parquet_path, engine="pyarrow")

    else:
        # Load precomputed embeddings
        try:
            embeddings_df = pd.read_parquet(parquet_path, engine="pyarrow")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Precomputed embeddings file not found at: {parquet_path}\n"
                f"Run with load_precomputed=False to generate the cache."
            )

    # Convert embeddings to numpy arrays if needed
    # When loaded from parquet, embeddings are object arrays containing individual vectors
    # We need to stack them into proper 2D arrays [num_patches, 128]
    first_emb = embeddings_df["image_embedding"].iloc[0]
    if isinstance(first_emb, np.ndarray) and first_emb.dtype == object:
        embeddings_df["image_embedding"] = embeddings_df["image_embedding"].apply(np.stack)
    elif not isinstance(first_emb, np.ndarray):
        embeddings_df["image_embedding"] = embeddings_df["image_embedding"].apply(np.array)

    return embeddings_df


def pil_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert a PIL Image to base64-encoded string.

    Takes a PIL Image object and encodes it as a base64 string
    suitable for use with OpenAI's vision API. The image is saved
    to an in-memory buffer and encoded without creating temporary files.

    Args:
        image: PIL Image object to convert
        format: Image format for encoding (default: "PNG")

    Returns:
        Base64-encoded string representation of the image

    Example:
        >>> from PIL import Image
        >>> img = Image.open("document.png")
        >>> base64_str = pil_image_to_base64(img)
        >>> # Use with OpenAI vision API
        >>> data_url = f"data:image/png;base64,{base64_str}"
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    base64_encoded = base64.b64encode(img_bytes)
    return base64_encoded.decode("ascii")


def display_retrieved_documents(
    results: list[tuple[str, float]],
    max_width: int = 350,
    height_per_image: int = 500,
) -> go.Figure:
    """
    Display retrieved document images with similarity scores.

    Creates a vertical stack of document images with their retrieval
    scores displayed as titles. Useful for visualizing RAG retrieval
    results before sending to a generation model.

    Args:
        results: List of (image_path, score) tuples from retrieval
        max_width: Maximum width in pixels for each image (maintains aspect ratio)
        height_per_image: Height allocation per image

    Returns:
        Plotly Figure object with the visualization

    Example:
        >>> results = retrieve_documents("What is AI?", top_k=3)
        >>> fig = display_retrieved_documents(results)
        >>> fig.show()
    """
    from plotly.subplots import make_subplots

    n_results = len(results)

    # Load and resize images
    resized_images = []
    for image_path, _ in results:
        img = Image.open(image_path)
        aspect_ratio = img.height / img.width
        new_width = min(max_width, img.width)
        new_height = int(new_width * aspect_ratio)
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_images.append(np.array(img_resized.convert("RGB")))

    # Generate subplot titles with scores
    subplot_titles = [f"Score: {score:.4f}" for _, score in results]

    # Create subplots in vertical layout
    fig = make_subplots(
        rows=n_results,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.02,
    )

    # Add image traces
    for i, img_array in enumerate(resized_images):
        fig.add_trace(go.Image(z=img_array), row=i + 1, col=1)

    # Update layout
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_layout(
        width=max_width + 100,
        height=height_per_image * n_results,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


def recreate_colpali_optimizations_collection(
    client,
    collection_name: str = "colpali-optimizations",
    embeddings_path: str = "ro_shared_data/pdfs/colpali-embeddings.parquet",
    muvera=None,
    allowed_docs=("MLS_C2_W1", "DLS_C1_W4"),
):
    """
    Recreate the colpali-optimizations collection from Lesson 3.

    Loads pre-computed optimized ColPali embeddings from ro_shared_data/pdfs/optimized/
    and creates a Qdrant collection with multiple optimization strategies: original,
    binary quantized, hierarchical pooling (2x, 4x), spatial pooling (row, column),
    and optionally MUVERA fixed-dimensional encodings.

    Uses the yield_optimized_embeddings() helper to load pre-computed optimizations
    for memory efficiency (processes one image at a time instead of loading all 1,737
    images into memory).

    Args:
        client: QdrantClient instance
        collection_name: Name for the collection (default: "colpali-optimizations")
        embeddings_path: DEPRECATED - No longer used. Pre-computed optimizations are
                        loaded from ro_shared_data/pdfs/optimized/ directory.
        muvera: Optional Muvera instance for creating MUVERA embeddings.
                If provided, adds "muvera_fde" named vector to the collection.
                MUVERA encodings are computed on-the-fly (not pre-computed).
        allowed_docs: Tuple of document name patterns to filter (default: only MLS_C2_W1
                     and DLS_C1_W4 documents, totaling ~100 images)

    Example:
        >>> from qdrant_client import QdrantClient
        >>> from fastembed.postprocess.muvera import Muvera
        >>> client = QdrantClient("http://localhost:6333")
        >>> muvera = Muvera(dim=128, k_sim=6, dim_proj=16, r_reps=20)
        >>> recreate_colpali_optimizations_collection(client, muvera=muvera)
    """
    from qdrant_client import models
    from tqdm import tqdm

    # Delete collection if exists
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    # Create collection with all optimization vectors
    vectors_config = {
        "original": models.VectorParams(
            size=128,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
            on_disk=True,
        ),
        "scalar_quantized": models.VectorParams(
            size=128,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                ),
            ),
            on_disk=True,
        ),
        "binary_quantized": models.VectorParams(
            size=128,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=True,
                ),
            ),
            on_disk=True,
        ),
        "hierarchical_2x": models.VectorParams(
            size=128,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
            on_disk=True,
        ),
        "hierarchical_4x": models.VectorParams(
            size=128,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
            on_disk=True,
        ),
        "row_pooled": models.VectorParams(
            size=128,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
            on_disk=True,
        ),
        "column_pooled": models.VectorParams(
            size=128,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
            on_disk=True,
        ),
    }

    # Add MUVERA vector config if muvera instance is provided
    if muvera is not None:
        # Calculate FDE size: 2^k_sim clusters × dim_proj × r_reps
        fde_size = (2**muvera.k_sim) * muvera.dim_proj * muvera.r_reps
        vectors_config["muvera_fde"] = models.VectorParams(
            size=fde_size,
            distance=models.Distance.DOT,
            on_disk=True,
            # No multivector config - single vector with HNSW enabled
        )

    client.create_collection(
        collection_name,
        vectors_config=vectors_config,
    )

    # Insert embeddings with all optimizations using pre-computed values
    imported_count = 0
    for i, (image_path, vectors) in enumerate(
        tqdm(
            yield_optimized_embeddings(load_precomputed=True, allowed_docs=allowed_docs),
            desc="Upserting embeddings",
        )
    ):
        imported_count += 1

        # Build vector dictionary with pre-computed optimizations
        vector_dict = {
            "original": vectors["original"],
            "scalar_quantized": vectors["original"],
            "binary_quantized": vectors["original"],
            "hierarchical_2x": vectors["hierarchical_2x"],
            "hierarchical_4x": vectors["hierarchical_4x"],
            "row_pooled": vectors["row_pooled"],
            "column_pooled": vectors["column_pooled"],
        }

        # Add MUVERA embedding if muvera instance is provided
        if muvera is not None:
            vector_dict["muvera_fde"] = muvera.process_document(vectors["original"])

        client.upsert(
            collection_name,
            points=[
                models.PointStruct(
                    id=i,
                    vector=vector_dict,
                    payload={
                        "image_path": image_path,
                    },
                )
            ],
        )

    print(f"✓ Collection '{collection_name}' created with {imported_count} points")


def compare_search_methods(
    baseline_search_fn,
    comparison_search_fn,
    baseline_name,
    comparison_name,
    query_text,
    limit=5,
    n_runs=10,
):
    """Compare two search methods by running queries N times.

    Executes baseline and comparison search functions multiple times to calculate
    performance metrics including timing statistics, precision, and speedup. Useful
    for benchmarking different retrieval approaches like ColPali vs MUVERA.

    Args:
        baseline_search_fn: Function that performs baseline search
            Must return tuple of (results, search_time)
        comparison_search_fn: Function that performs comparison search
            Must return tuple of (results, search_time)
        baseline_name: Display name for baseline method (e.g., "ColPali")
        comparison_name: Display name for comparison method (e.g., "MUVERA")
        query_text: The query text for display
        limit: Number of results to return
        n_runs: Number of times to run each query (default: 10)

    Returns:
        Dictionary with comparison metrics:
        - baseline_avg_time, baseline_median_time
        - comparison_avg_time, comparison_median_time
        - avg_speedup, median_speedup
        - precision (overlap ratio)
        - baseline_results, comparison_results

    Example:
        >>> result = compare_search_methods(
        ...     baseline_search_fn=lambda: search_colpali(query_emb, limit=5),
        ...     comparison_search_fn=lambda: search_muvera(query_emb, limit=5),
        ...     baseline_name="ColPali",
        ...     comparison_name="MUVERA",
        ...     query_text="coffee mug",
        ...     limit=5,
        ...     n_runs=10
        ... )
    """
    # Run baseline search N times
    baseline_times = []
    baseline_results = None
    for _ in range(n_runs):
        results, search_time = baseline_search_fn()
        baseline_times.append(search_time)
        baseline_results = results

    # Run comparison search N times
    comparison_times = []
    comparison_results = None
    for _ in range(n_runs):
        results, search_time = comparison_search_fn()
        comparison_times.append(search_time)
        comparison_results = results

    # Calculate timing statistics
    baseline_avg_time = np.mean(baseline_times)
    baseline_median_time = np.median(baseline_times)
    comparison_avg_time = np.mean(comparison_times)
    comparison_median_time = np.median(comparison_times)

    # Extract result IDs
    baseline_ids = [r.id for r in baseline_results[:limit]]
    comparison_ids = [r.id for r in comparison_results[:limit]]

    # Calculate metrics
    overlap = len(set(baseline_ids) & set(comparison_ids))
    precision = overlap / limit
    avg_speedup = (
        baseline_avg_time / comparison_avg_time if comparison_avg_time > 0 else 0
    )
    median_speedup = (
        baseline_median_time / comparison_median_time
        if comparison_median_time > 0
        else 0
    )

    # Print comparison
    print(f"\nQuery: '{query_text}' (ran {n_runs} times)")
    print(
        f"  {baseline_name} avg time: {baseline_avg_time * 1000:.2f}ms "
        f"(median: {baseline_median_time * 1000:.2f}ms)"
    )
    print(
        f"  {comparison_name} avg time:  {comparison_avg_time * 1000:.2f}ms "
        f"(median: {comparison_median_time * 1000:.2f}ms)"
    )
    print(f"  Speedup (avg): {avg_speedup:.1f}x faster")
    print(f"  Speedup (median): {median_speedup:.1f}x faster")
    print(f"  Precision@{limit}: {precision:.1%} ({overlap}/{limit} overlap)")

    # Visualize results if payload with image_path is available
    if (
        baseline_results
        and len(baseline_results) > 0
        and baseline_results[0].payload
        and "image_path" in baseline_results[0].payload
    ):
        # Extract image paths and scores for visualization
        baseline_vis = [
            (point.payload["image_path"], point.score)
            for point in baseline_results[:limit]
        ]
        comparison_vis = [
            (point.payload["image_path"], point.score)
            for point in comparison_results[:limit]
        ]

        # Display baseline results
        print(f"\n{baseline_name} retrieved:")
        fig = display_retrieved_documents(baseline_vis)
        fig.show()

        # Display comparison results
        print(f"\n{comparison_name} retrieved:")
        fig = display_retrieved_documents(comparison_vis)
        fig.show()

    return {
        "baseline_avg_time": baseline_avg_time,
        "baseline_median_time": baseline_median_time,
        "comparison_avg_time": comparison_avg_time,
        "comparison_median_time": comparison_median_time,
        "avg_speedup": avg_speedup,
        "median_speedup": median_speedup,
        "precision": precision,
        "baseline_results": baseline_results,
        "comparison_results": comparison_results,
    }

