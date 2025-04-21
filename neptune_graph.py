#!/usr/bin/env python3

import os
import boto3
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from gremlin_python import statics
from gremlin_python.structure.graph import Graph
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.strategies import *
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.traversal import T
from IPython.display import display, HTML
import ipycytoscape

class NeptuneGraphManager:
    def __init__(self, neptune_endpoint=None, neptune_port=8182, region=None):
        """Initialize Neptune connection manager."""
        self.neptune_endpoint = neptune_endpoint or os.getenv('NEPTUNE_ENDPOINT')
        self.neptune_port = neptune_port
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')
        
        if not self.neptune_endpoint:
            raise ValueError(
                "Neptune endpoint is required. Please provide it either:\n"
                "1. As an argument: NeptuneGraphManager(neptune_endpoint='your-endpoint')\n"
                "2. Through environment variable: export NEPTUNE_ENDPOINT='your-endpoint'"
            )
        
        # Initialize Gremlin connection
        self.graph = Graph()
        self.connection = DriverRemoteConnection(
            f'wss://{self.neptune_endpoint}:{self.neptune_port}/gremlin',
            'g'
        )
        self.g = self.graph.traversal().withRemote(self.connection)
        
        # Initialize visualization settings
        plt.style.use('seaborn')
        self.visualization_settings = {
            'node_size': 1000,
            'font_size': 8,
            'edge_width': 1.5,
            'fig_size': (12, 8)
        }

    def store_triples(self, triples):
        """Store triples in Neptune graph database."""
        print(f"Storing {len(triples)} triples in Neptune...")
        
        for triple in triples:
            try:
                # Add subject vertex if it doesn't exist
                subject = self.g.V().has('name', triple['subject']).fold().coalesce(
                    __.unfold(),
                    __.addV('entity').property('name', triple['subject'])
                ).next()
                
                # Add object vertex if it doesn't exist
                obj = self.g.V().has('name', triple['object']).fold().coalesce(
                    __.unfold(),
                    __.addV('entity').property('name', triple['object'])
                ).next()
                
                # Add edge between subject and object
                self.g.V(subject).addE(triple['predicate']).to(__.V(obj)).next()
                
            except Exception as e:
                print(f"Error storing triple {triple}: {str(e)}")
                continue
        
        print("Triples stored successfully!")

    def get_graph_data(self, limit=100):
        """Retrieve graph data from Neptune."""
        print("Retrieving graph data from Neptune...")
        
        # Get vertices and edges
        vertices = self.g.V().limit(limit).valueMap(True).toList()
        edges = self.g.E().limit(limit).valueMap(True).toList()
        
        # Convert to NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for v in vertices:
            node_id = str(v['id'][0])
            node_name = v['name'][0] if 'name' in v else 'Unknown'
            G.add_node(node_id, name=node_name)
        
        # Add edges
        for e in edges:
            source = str(e['id'][0].split('->')[0])
            target = str(e['id'][0].split('->')[1])
            label = list(e.keys())[0]  # Get the edge label
            G.add_edge(source, target, label=label)
        
        return G

    def visualize_graph(self, G=None, output_file=None):
        """Visualize the graph using matplotlib."""
        if G is None:
            G = self.get_graph_data()
        
        plt.figure(figsize=self.visualization_settings['fig_size'])
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=self.visualization_settings['node_size'],
            node_color='lightblue',
            alpha=0.7
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            width=self.visualization_settings['edge_width'],
            arrows=True,
            arrowsize=20
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=self.visualization_settings['font_size']
        )
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=self.visualization_settings['font_size']
        )
        
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Graph saved to {output_file}")
        else:
            plt.show()

    def visualize_interactive(self, G=None):
        """Create an interactive visualization using ipycytoscape."""
        if G is None:
            G = self.get_graph_data()
        
        # Convert NetworkX graph to cytoscape format
        cytoscape_graph = ipycytoscape.CytoscapeWidget()
        cytoscape_graph.graph.add_graph_from_networkx(G)
        
        # Set layout
        cytoscape_graph.set_layout(name='cose')
        
        # Set style
        style = {
            'selector': 'node',
            'style': {
                'content': 'data(name)',
                'text-valign': 'center',
                'text-halign': 'center',
                'background-color': '#11479e',
                'text-outline-color': '#11479e',
                'text-outline-width': 2,
                'color': '#fff',
                'width': 80,
                'height': 80
            }
        }
        cytoscape_graph.set_style([style])
        
        return cytoscape_graph

    def query_graph(self, query):
        """Execute a Gremlin query on the graph."""
        try:
            result = self.g.V().has('name', query).bothE().bothV().path().by('name').toList()
            return result
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return []

    def close(self):
        """Close the Neptune connection."""
        if self.connection:
            self.connection.close()
            print("Neptune connection closed.")

def main():
    """Example usage of NeptuneGraphManager."""
    try:
        # Initialize Neptune manager
        neptune_manager = NeptuneGraphManager()
        
        # Example triples
        example_triples = [
            {'subject': 'albert einstein', 'predicate': 'developed', 'object': 'theory of relativity'},
            {'subject': 'albert einstein', 'predicate': 'received', 'object': 'nobel prize in physics'},
            {'subject': 'theory of relativity', 'predicate': 'is part of', 'object': 'modern physics'}
        ]
        
        # Store triples
        neptune_manager.store_triples(example_triples)
        
        # Get and visualize graph
        G = neptune_manager.get_graph_data()
        neptune_manager.visualize_graph(G, output_file='knowledge_graph.png')
        
        # Create interactive visualization
        interactive_graph = neptune_manager.visualize_interactive(G)
        display(interactive_graph)
        
        # Example query
        results = neptune_manager.query_graph('albert einstein')
        print("\nQuery results for 'albert einstein':")
        for path in results:
            print(path)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'neptune_manager' in locals():
            neptune_manager.close()

if __name__ == "__main__":
    main() 