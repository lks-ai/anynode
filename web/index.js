import { app } from '../../scripts/app.js';
import { loadCSS, loadScript, sleep } from './utils.js';
import { applyStyles } from './node-styles.js';
import { showCode } from './show-code.js';
import { storeAnyNodeOutputs } from './store-outputs.js';

app.registerExtension({
  name: 'AnyNode',
  async init() {
    const CDN_BASE_URL = 'https://cdnjs.cloudflare.com/ajax/libs';
    await loadScript(`${CDN_BASE_URL}/highlight.js/11.9.0/highlight.min.js`);
    await loadScript(`${CDN_BASE_URL}/highlightjs-line-numbers.js/2.8.0/highlightjs-line-numbers.min.js`);
    await loadCSS(`${CDN_BASE_URL}/highlight.js/11.9.0/styles/atom-one-dark.min.css`);
    await loadCSS(`/extensions/anynode/css/show-code.css`);
    hljs.configure({ ignoreUnescapedHTML: true });
  },
  async nodeCreated(node) {
    await sleep(0); // Wait for node object to be updated
    if (node.type.includes('AnyNode')) {
      applyStyles(node);
      node.type === 'AnyNodeShowCode' ? showCode(node) : storeAnyNodeOutputs(node);
    }
  },
});
