import jax, jax.numpy as jnp
from jaxlib import xla_client as xc
from graphviz import Source
from IPython.display import SVG, display, Image, HTML
import re
import html
import numpy as np

def bytes_str(bytes):
    if abs(bytes) < 1024:
        return f"{bytes} B"
    elif abs(bytes) < 1024**2:
        return f"{bytes / 1024:.1f} kB"
    elif abs(bytes) < 1024**3:
        return f"{bytes / 1024**2:.1f} MB"
    else:
        return f"{bytes / 1024**3:.1f} GB"

def print_memory_usage(fcompiled, show_host_mem=False):
    m = fcompiled.memory_analysis()

    if m.generated_code_size_in_bytes > 1024*1024:
        print("Warning! We have constant folding!")

    print(f"code  : {bytes_str(m.generated_code_size_in_bytes )}")
    print(f"temp  : {bytes_str(m.temp_size_in_bytes)}")
    print(f"arg   : {bytes_str(m.argument_size_in_bytes)}")
    print(f"output: {bytes_str(m.output_size_in_bytes)}")
    print(f"alias : {bytes_str(-m.alias_size_in_bytes)}")
    print(f"peak  : {bytes_str(m.peak_memory_in_bytes )}")

    if show_host_mem:
        print(f"host code : {bytes_str(m.host_generated_code_size_in_bytes )}")
        print(f"host temp : {bytes_str(m.host_temp_size_in_bytes)}")
        print(f"host arg  : {bytes_str(m.host_argument_size_in_bytes)}")
        print(f"host output : {bytes_str(m.host_output_size_in_bytes)}")
        print(f"host alias : {bytes_str(m.host_alias_size_in_bytes)}")

def hlo_to_svg_text(hlo_text: str, title: str = None):
    # Parse HLO text -> XLA HloModule, then emit DOT and render to SVG
    mod = xc._xla.hlo_module_from_text(hlo_text)        # private API
    dot = xc._xla.hlo_module_to_dot_graph(mod)          # private API
    svg_bytes = Source(dot).pipe(format="svg")

    # svg_bytes is bytes; convert to text so we can optionally inject a <title>
    if isinstance(svg_bytes, bytes):
        svg_text = svg_bytes.decode("utf-8")
    else:
        svg_text = str(svg_bytes)

    if title:
        esc = html.escape(title)

        title_fragment = (
            f"<title>{title}</title>"
            f"<g id=\"hlo_title\">"
            f"<text x=\"22\" y=\"20\" font-family=\"sans-serif\" "
            f"font-size=\"16\" font-weight=\"bold\" fill=\"#1976d2\">{title}</text>"
            f"</g>"
        )
        if re.search(r'</svg\s*>', svg_text, flags=re.IGNORECASE):
            svg_text = re.sub(r'</svg\s*>', title_fragment + '</svg>', svg_text, count=1, flags=re.IGNORECASE)
        else:
            svg_text = title_fragment + svg_text

    return svg_text

def resize_svg(svg, width=400):
    """Return an SVG string with width fixed to `width` and height auto."""
    if hasattr(svg, "data"):          # IPython.display.SVG
        svg_text = svg.data
    elif isinstance(svg, bytes):
        svg_text = svg.decode("utf-8")
    else:
        svg_text = str(svg)

    # remove any existing width/height on the <svg> tag
    svg_text = re.sub(r'(<svg[^>]*?)\swidth="[^"]*"', r'\1', svg_text, count=1)
    svg_text = re.sub(r'(<svg[^>]*?)\sheight="[^"]*"', r'\1', svg_text, count=1)
    # add our width; let height be computed from viewBox
    svg_text = re.sub(r'<svg', f'<svg width="{width}px"', svg_text, count=1)
    return SVG(svg_text)

def show_hlo_info(f, *args, mode="mem_post", width=400, save=False, show_host_mem=False, **kwargs):
    lo = f.lower(*args, **kwargs)
    title = f.__name__

    comp = lo.compile()

    if "mem" in mode:
        print(f"--------  Memory usage of {title}  ---------")
        print_memory_usage(comp, show_host_mem=show_host_mem)
    if "pre" in mode:
        pre_hlo  = lo.as_text(dialect="hlo")
        svg = hlo_to_svg_text(pre_hlo, title=f"{title} (pre)")
        if save:
            with open(f"{title}_pre.svg", "w") as f:
                f.write(svg)
        else:
            display(resize_svg(svg, width=width))
    if "post" in mode:
        m = comp.memory_analysis()
        mem_fac = m.temp_size_in_bytes / np.maximum(m.argument_size_in_bytes, 1)
        title_post = f"{title} (tmp: {bytes_str(m.temp_size_in_bytes)}, {mem_fac:.2g}x)"

        post_hlo = comp.as_text()
        svg = hlo_to_svg_text(post_hlo, title=title_post)
        if save:
            with open(f"{title}_post.svg", "w") as f:
                f.write(svg)
        else:
            display(resize_svg(svg, width=width))