class NodeAware:
    """Utility class for working with ComfyUI workflow data from hidden `prompt` in your custom nodes inputs"""
    def __init__(self, workflow=None, pnginfo=None):
        if pnginfo:
            if not isinstance(pnginfo, list):
                if isinstance(pnginfo, dict):
                    workflow = pnginfo["workflow"]
            else:
                workflow = pnginfo[0]["workflow"]            
        self.workflow = workflow
    
    def get_node(self, node_id):
        return next((node for node in self.workflow['nodes'] if node['id'] == node_id), None)

    def get_connected_nodes(self, node_id, direction='both'):
        """Find all connected nodes and what slots they are connected to"""
        connections = {'inputs': [], 'outputs': []}
        for link in self.workflow['links']:
            if direction in ('both', 'inputs') and link[3] == node_id:
                source_node = self.get_node(link[1])
                connections['inputs'].append({
                    'source_node': link[1],
                    'source_node_name': source_node['type'],
                    'source_slot': link[2],
                    'target_node': link[3],
                    'target_slot': link[4],
                    'type': link[5]
                })
            if direction in ('both', 'outputs') and link[1] == node_id:
                target_node = self.get_node(link[3])
                connections['outputs'].append({
                    'target_node': link[3],
                    'target_node_name': target_node['type'],
                    'target_slot': link[4],
                    'source_node': link[1],
                    'source_slot': link[2],
                    'type': link[5]
                })
        return connections

    def summarize_connections(self, node_id):
        """Summarize the connections so that an LLM can have situational context for it's system prompt"""
        connections = self.get_connected_nodes(node_id)
        summary = f"### Node {node_id} Connections\n\n"
        summary += "#### Inputs\n"
        for input_conn in connections['inputs']:
            summary += f"- From Node {input_conn['source_node']} ({input_conn['source_node_name']}) on slot {input_conn['source_slot']} (type: {input_conn['type']})\n"

        summary += "\n#### Outputs\n"
        for output_conn in connections['outputs']:
            summary += f"- To Node {output_conn['target_node']} ({output_conn['target_node_name']}) on slot {output_conn['target_slot']} (type: {output_conn['type']})\n"

        return summary
    
    def find_nodes(self, **kwargs):
        """Search for nodes by id, type, input_types:list, or output_types:list. Returns a *list of the node objects from the workflow"""
        print(f"\nFinding Nodes in Workspace. kwargs: {kwargs}")
        result = []
        node_id = kwargs.get('id')
        node_type = kwargs.get('type')
        input_types = kwargs.get('input_types', [])
        output_types = kwargs.get('output_types', [])

        for node in self.workflow['nodes']:
            if node_id and int(node['id']) == int(node_id):
                result.append(node)
            elif node_type and node['type'] == node_type:
                result.append(node)
            elif input_types:
                for input_link in node['inputs']:
                    if input_link['type'] in input_types:
                        result.append(node)
                        break
            elif output_types:
                for output_link in node['outputs']:
                    if output_link['type'] in output_types:
                        result.append(node)
                        break

        return None if len(result) == 0 else result
    
    def find_node(self, **kwargs):
        r = self.find_nodes(**kwargs)
        if r:
            return r[0]
        return r

if __name__ == "__main__":
    # Example usage
    workflow_data = {'last_node_id': 13, 'last_link_id': 13, 'nodes': [{'id': 5, 'type': 'ShowText|pysssss', 'pos': [1036.8042600016947, 84.57110116501482], 'size': [305.0083925861959, 219.76858999709455], 'flags': {}, 'order': 7, 'mode': 0, 'inputs': [{'name': 'text', 'type': 'STRING', 'link': 1, 'widget': {'name': 'text'}}], 'outputs': [{'name': 'STRING', 'type': 'STRING', 'links': None, 'shape': 6}], 'properties': {'Node name for S&R': 'ShowText|pysssss'}, 'widgets_values': ['', 'Input 1 -> Type: dict, Keys: samples\nInput 2 -> Type: Tensor']}, {'id': 3, 'type': 'LoadVideo [n-suite]', 'pos': [214, 64], 'size': [210, 630], 'flags': {}, 'order': 0, 'mode': 0, 'outputs': [{'name': 'IMAGES', 'type': 'IMAGE', 'links': [5], 'shape': 6, 'slot_index': 0}, {'name': 'EMPTY LATENTS', 'type': 'LATENT', 'links': None, 'shape': 6}, {'name': 'METADATA', 'type': 'STRING', 'links': [], 'shape': 3, 'slot_index': 2}, {'name': 'WIDTH', 'type': 'INT', 'links': [], 'shape': 3, 'slot_index': 3}, {'name': 'HEIGHT', 'type': 'INT', 'links': None, 'shape': 3}, {'name': 'META_FPS', 'type': 'INT', 'links': [], 'shape': 3, 'slot_index': 5}, {'name': 'META_N_FRAMES', 'type': 'INT', 'links': None, 'shape': 3}], 'properties': {'Node name for S&R': 'LoadVideo [n-suite]'}, 'widgets_values': ['swan-lake-tchaikovski.mp4', '/view?filename=swan-lake-tchaikovski.mp4&type=input&subfolder=n-suite', 'original', 'none', 512, 0, 0, 0, True, 'image', None]}, {'id': 8, 'type': 'CheckpointLoaderSimple', 'pos': [-333, -268], 'size': {'0': 315, '1': 98}, 'flags': {}, 'order': 1, 'mode': 0, 'outputs': [{'name': 'MODEL', 'type': 'MODEL', 'links': [7], 'shape': 3}, {'name': 'CLIP', 'type': 'CLIP', 'links': [8, 9], 'shape': 3, 'slot_index': 1}, {'name': 'VAE', 'type': 'VAE', 'links': None, 'shape': 3}], 'properties': {'Node name for S&R': 'CheckpointLoaderSimple'}, 'widgets_values': ['3dMixCharacter_v20Realism.safetensors']}, {'id': 10, 'type': 'CLIPTextEncode', 'pos': [-149, 17], 'size': [250, 88], 'flags': {}, 'order': 4, 'mode': 0, 'inputs': [{'name': 'clip', 'type': 'CLIP', 'link': 9}], 'outputs': [{'name': 'CONDITIONING', 'type': 'CONDITIONING', 'links': [11], 'shape': 3, 'slot_index': 0}], 'properties': {'Node name for S&R': 'CLIPTextEncode'}, 'widgets_values': ['']}, {'id': 9, 'type': 'CLIPTextEncode', 'pos': [-142, -104], 'size': [234, 78], 'flags': {}, 'order': 3, 'mode': 0, 'inputs': [{'name': 'clip', 'type': 'CLIP', 'link': 8}], 'outputs': [{'name': 'CONDITIONING', 'type': 'CONDITIONING', 'links': [10], 'shape': 3, 'slot_index': 0}], 'properties': {'Node name for S&R': 'CLIPTextEncode'}, 'widgets_values': ['a cat']}, {'id': 7, 'type': 'KSampler', 'pos': [299, -230], 'size': {'0': 315, '1': 262}, 'flags': {}, 'order': 5, 'mode': 0, 'inputs': [{'name': 'model', 'type': 'MODEL', 'link': 7, 'slot_index': 0}, {'name': 'positive', 'type': 'CONDITIONING', 'link': 10}, {'name': 'negative', 'type': 'CONDITIONING', 'link': 11}, {'name': 'latent_image', 'type': 'LATENT', 'link': 12, 'slot_index': 3}], 'outputs': [{'name': 'LATENT', 'type': 'LATENT', 'links': [6], 'shape': 3, 'slot_index': 0}], 'properties': {'Node name for S&R': 'KSampler'}, 'widgets_values': [583656379821656, 'randomize', 20, 8, 'euler', 'normal', 1]}, {'id': 11, 'type': 'EmptyLatentImage', 'pos': [-265, 183], 'size': {'0': 315, '1': 106}, 'flags': {}, 'order': 2, 'mode': 0, 'outputs': [{'name': 'LATENT', 'type': 'LATENT', 'links': [12], 'shape': 3}], 'properties': {'Node name for S&R': 'EmptyLatentImage'}, 'widgets_values': [512, 512, 1]}, {'id': 4, 'type': 'AnyNode', 'pos': [650, -185], 'size': {'0': 400, '1': 200}, 'flags': {}, 'order': 6, 'mode': 0, 'inputs': [{'name': 'any', 'type': '*', 'link': 6}, {'name': 'any2', 'type': '*', 'link': 5}], 'outputs': [{'name': 'any', 'type': '*', 'links': [1, 13], 'shape': 3, 'slot_index': 0}], 'properties': {'Node name for S&R': 'AnyNode'}, 'widgets_values': ["Output the type and some information about the inputs.\nOutput should be a string.\n\nOutput should have class information.\nIf it's a dict, then output the dict keys.\nIf it's a Tensor, output the shape.\netc.", 'gpt-4o']}, {'id': 13, 'type': 'AnyNodeCodeViewer', 'pos': [734, 129], 'size': {'0': 210, '1': 26}, 'flags': {}, 'order': 8, 'mode': 0, 'inputs': [{'name': 'ctrl', 'type': 'DICT', 'link': 13}], 'outputs': [{'name': 'DICT', 'type': 'DICT', 'links': None, 'shape': 3, 'slot_index': 0}], 'properties': {'Node name for S&R': 'AnyNodeCodeViewer'}}], 'links': [[1, 4, 0, 5, 0, 'STRING'], [5, 3, 0, 4, 1, '*'], [6, 7, 0, 4, 0, '*'], [7, 8, 0, 7, 0, 'MODEL'], [8, 8, 1, 9, 0, 'CLIP'], [9, 8, 1, 10, 0, 'CLIP'], [10, 9, 0, 7, 1, 'CONDITIONING'], [11, 10, 0, 7, 2, 'CONDITIONING'], [12, 11, 0, 7, 3, 'LATENT'], [13, 4, 0, 13, 0, 'DICT']], 'groups': [], 'config': {}, 'extra': {}, 'version': 0.4}

    node_id = 13  # Example node ID
    node_aware = NodeAware(workflow_data)
    connections = node_aware.get_connected_nodes(node_id)
    summary = node_aware.summarize_connections(node_id)

    print("Connections:", connections)
    print("Summary:\n", summary)

    found_nodes = node_aware.find_node(id=node_id)
    print("Found Node:", found_nodes)