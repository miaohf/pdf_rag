import { visit, SKIP } from 'unist-util-visit'

// Rehype 插件：按搜索词高亮文本
// 使用方式：rehypePlugins: [[rehypeHighlightSearch, { searchTerm }]]
export default function rehypeHighlightSearch(options?: { searchTerm?: string }) {
  const term = (options?.searchTerm || '').trim()
  if (!term) {
    // 返回一个空transformer
    return () => {}
  }

  // 预编译正则
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const regex = new RegExp(escaped, 'gi')

  const SKIP_PARENTS = new Set(['code', 'pre', 'script', 'style', 'noscript'])

  return (tree: any) => {
    visit(tree, 'text', (node: any, index: number | undefined, parent: any) => {
      if (typeof index !== 'number' || !parent) return
      if (!node || typeof node.value !== 'string' || node.value.length === 0) return

      // 跳过不应高亮的父节点
      if (parent.type === 'element' && SKIP_PARENTS.has(parent.tagName)) {
        return
      }

      const value: string = node.value
      if (!regex.test(value)) {
        return
      }

      // 重置正则状态，避免后续test失败
      regex.lastIndex = 0

      const parts = value.split(regex)
      const matches = value.match(regex) || []

      const newChildren: any[] = []
      for (let i = 0; i < parts.length; i++) {
        const textPart = parts[i]
        if (textPart) {
          newChildren.push({ type: 'text', value: textPart })
        }
        if (i < parts.length - 1) {
          const m = matches[i] || ''
          newChildren.push({
            type: 'element',
            tagName: 'mark',
            properties: {
              className: 'search-highlight',
              style: 'color: #1f2937 !important; font-weight: 600;'
            },
            children: [{ type: 'text', value: m }],
          })
        }
      }

      // 用新children替换当前text节点
      parent.children.splice(index, 1, ...newChildren)

      // 跳过刚插入的子节点
      return SKIP
    })
  }
} 