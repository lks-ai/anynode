export function storeAnyNodeOutputs(node) {
  const onExecuted = node.onExecuted;
  node.onExecuted = function (event) {
    onExecuted && onExecuted.apply(this, arguments);
    node.outputs_values = node.outputs_values || [];
    node.outputs.forEach((output, index) => {
      node.outputs_values[index] = event[output.name] && event[output.name][0] || undefined;
    });
  };
}
