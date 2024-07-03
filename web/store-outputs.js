export function storeAnyNodeOutputs(node) {
  node.onExecuted = function (event) {
    node.outputs_values = node.outputs_values || [];
    node.outputs.forEach((output, index) => {
      node.outputs_values[index] = event[output.name] && event[output.name][0] || undefined;
    });
  };
}
