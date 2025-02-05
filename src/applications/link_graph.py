import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import powerlaw
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import create_folder_if_not_exists, save_json

# Global variable to store the results
analysis_results = {}

# Add edges based on similarity
def build_graph_from_embeddings(df, threshold=0.85):
    G = nx.Graph()

    for index, row in df.iterrows():
        G.add_node(
            row['ID'], 
            url=row["News URL"], 
            embedding=embeddings[row['ID']], 
            category=row["News Category"]  # Add category as an attribute
        )

    similarity_matrix = cosine_similarity(embeddings)

    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    return G

def draw_graph(graph, categories, output_path="graph_embeddings.png"):
    """
    Generates a graph visualization with colors for categories.
    
    Args:
        graph: NetworkX graph.
        categories: List with node categories (ordered by ID).
        output_path: Path to save the visualization.
    """
    # Layout for node positioning
    pos = nx.spring_layout(graph, k=0.3, seed=42)  

    # Generate colors for categories
    unique_categories = list(set(categories))
    category_colors = {cat: plt.cm.tab20(i / len(unique_categories)) for i, cat in enumerate(unique_categories)}
    node_colors = [category_colors[graph.nodes[node]['category']] for node in graph.nodes()]

    # Edge weights
    weights = [graph[u][v]['weight'] for u, v in graph.edges()]

    # Draw the graph using draw_networkx
    plt.figure(figsize=(12, 12))
    nx.draw_networkx(
        graph, pos,
        with_labels=False,  # Remove node labels
        node_color=node_colors,
        node_size=10,
        font_size=8,
        font_weight='bold',
        edge_color=weights,
        edge_cmap=plt.cm.Blues,
        width=0.1
    )

    # Add legend for categories
    legend_handles = [
        plt.Line2D([], [], marker='o', color=color, linestyle='None', markersize=10, label=cat)
        for cat, color in category_colors.items()
    ]
    plt.legend(handles=legend_handles, loc='best', title="Categories")
    plt.title("Embedding Graph Visualization by Category")
    plt.savefig(output_path)  # Save the graph to a file
    plt.close()

def get_top_k(metric_dict, k=5):
    """Function to get the top k nodes based on a metric."""
    return sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:k]

def plot_histogram(values, title, xlabel, ylabel, path_save, filename):
    """Function to plot histogram with vertical line at the mean."""
    mean_value = np.mean(values)
    
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    # Display the mean value on the graph
    plt.text(mean_value + 0.01, plt.ylim()[1] * 0.9, f'Mean: {mean_value:.4f}', color='red', fontsize=12)

    plt.savefig(f"{path_save}/{filename}")
    plt.close()

def degree_centrality_analysis(G, path_save, top_k=5):
    degree_centrality = nx.degree_centrality(G)
    top_5_degree_centrality = get_top_k(degree_centrality, top_k)
    
    analysis_results['degree_centrality'] = {
        'top_5': top_5_degree_centrality,
        'histogram': f"{path_save}/degree_centrality_histogram.png"
    }
    
    degree_centrality_values = list(degree_centrality.values())

    plot_histogram(degree_centrality_values, 'Degree Centrality Histogram', 'Degree Centrality', 'Number of Nodes', path_save, 'degree_centrality_histogram.png')

def betweenness_centrality_analysis(G, path_save, top_k=5):
    betweenness = nx.betweenness_centrality(G, weight='weight')
    top_5_betweenness = get_top_k(betweenness, top_k)
    
    analysis_results['betweenness_centrality'] = {
        'top_5': top_5_betweenness,
        'histogram': f"{path_save}/betweenness_centrality_histogram.png"
    }
    
    betweenness_values = list(betweenness.values())
    plot_histogram(betweenness_values, 'Betweenness Centrality Histogram', 'Betweenness Centrality', 'Number of Nodes', path_save, 'betweenness_centrality_histogram.png')

def closeness_centrality_analysis(G, path_save, top_k=5):
    closeness = nx.closeness_centrality(G)
    top_5_closeness = get_top_k(closeness, top_k)
    
    analysis_results['closeness_centrality'] = {
        'top_5': top_5_closeness,
        'histogram': f"{path_save}/closeness_centrality_histogram.png"
    }
    
    closeness_values = list(closeness.values())
    plot_histogram(closeness_values, 'Closeness Centrality Histogram', 'Closeness Centrality', 'Number of Nodes', path_save, 'closeness_centrality_histogram.png')

def clustering_analysis(G, path_save, top_k=5):
    clustering = nx.clustering(G)
    top_5_clustering = get_top_k(clustering, top_k)
    
    analysis_results['clustering_coefficient'] = {
        'top_5': top_5_clustering,
        'histogram': f"{path_save}/clustering_histogram.png"
    }
    
    clustering_values = list(clustering.values())
    plot_histogram(clustering_values, 'Clustering Coefficient Histogram', 'Clustering Coefficient', 'Number of Nodes', path_save, 'clustering_histogram.png')

def network_density_analysis(G):
    density = nx.density(G)
    analysis_results['network_density'] = density

def check_power_law_and_plot(graph, path_save):
    """
    Check if the degree distribution of a graph follows a power-law distribution.
    Plot the degree distribution and save the results.
    
    Args:
        graph: NetworkX graph.
        path_save: Path to save the plots and results.
    """
    # Get degree values
    degree_sequence = [degree for _, degree in graph.degree()]
    degree_sequence = np.array(degree_sequence)
    
    # Filter out nodes with degree 0 to avoid log(0) issues
    degree_sequence = degree_sequence[degree_sequence > 0]
    
    # Fit the degree distribution to a power-law
    try:
        fit = powerlaw.Fit(degree_sequence)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        result = fit.distribution_compare('power_law', 'exponential')
        
        print(f"Power-law fit results: alpha={alpha:.2f}, xmin={xmin}")
        print(f"Comparison with exponential: Log-likelihood ratio={result[0]:.2f}, p-value={result[1]:.2f}")
        
        # Save the analysis results
        with open(f"{path_save}/power_law_analysis.txt", "w") as f:
            f.write(f"Power-law fit results:\n")
            f.write(f"Alpha: {alpha:.2f}\n")
            f.write(f"Xmin: {xmin}\n")
            f.write(f"Comparison with exponential:\n")
            f.write(f"Log-likelihood ratio: {result[0]:.2f}\n")
            f.write(f"P-value: {result[1]:.2f}\n")
        
        # Plot the degree distribution
        plt.figure(figsize=(8, 6))
        fit.plot_pdf(color='blue', linewidth=2, label='Empirical data')
        fit.power_law.plot_pdf(color='red', linestyle='--', linewidth=2, label='Power-law fit')
        plt.xlabel("Degree")
        plt.ylabel("Probability")
        plt.title("Degree Distribution and Power-law Fit")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{path_save}/power_law_fit.png")
        plt.close()
    
    except Exception as e:
        print(f"Error fitting power-law distribution: {e}")


    # Verificação da aderência à lei de potência
    analysis_results['power_law'] = {
        "alpha": alpha,
        "plot": f"{path_save}/power_law_fit.png"
    }

# Dataset and embeddings paths
university = "UNESP"
data_path = f"data/postprocessing/{university}/data.json"
embeddings_path = f"data/embeddings/{university}/multilingual-e5-base/Text_embeddings.npy"

path_save = f"results/graph/{university}"
create_folder_if_not_exists(path_save)

df = pd.read_json(data_path)
embeddings = np.load(embeddings_path)

# Build the graph
G = build_graph_from_embeddings(df)

# List of categories for the nodes
categories = [category.lower() for category in df['News Category'].tolist()]

# Generate graph visualization with categories
draw_graph(G, categories, output_path=f"{path_save}/graph_embeddings_category.png")

# Graph analysis
analysis_results['graph_info'] = {
    'nodes': G.number_of_nodes(),
    'edges': G.number_of_edges()
}

# Complex network metrics analysis
check_power_law_and_plot(G, path_save)
degree_centrality_analysis(G, path_save)
betweenness_centrality_analysis(G, path_save)
closeness_centrality_analysis(G, path_save)
clustering_analysis(G, path_save)
network_density_analysis(G)

save_json(analysis_results, f"{path_save}/network_analysis_results.json")

