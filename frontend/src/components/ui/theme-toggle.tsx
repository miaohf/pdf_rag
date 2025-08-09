'use client'

import { Sun, Moon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useTheme } from '@/lib/theme'
import { useState, useEffect } from 'react'

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    // 避免水合错误，在客户端渲染前不显示具体图标
    return (
      <Button variant="ghost" size="icon" className="w-9 h-9">
        <div className="w-4 h-4" />
        <span className="sr-only">切换主题</span>
      </Button>
    )
  }

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light')
  }

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={toggleTheme}
      className="w-9 h-9"
    >
      {theme === 'light' ? (
        <Sun className="w-4 h-4" />
      ) : (
        <Moon className="w-4 h-4" />
      )}
      <span className="sr-only">切换到{theme === 'light' ? '深色' : '浅色'}主题</span>
    </Button>
  )
} 