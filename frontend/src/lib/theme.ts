'use client'

import { useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

export function useTheme() {
  const [theme, setTheme] = useState<Theme>('light')
  const [resolvedTheme, setResolvedTheme] = useState<'light' | 'dark'>('light')

  useEffect(() => {
    // 从localStorage读取保存的主题设置
    const savedTheme = localStorage.getItem('theme') as Theme | null
    if (savedTheme) {
      setTheme(savedTheme)
    }
  }, [])

  useEffect(() => {
    const root = window.document.documentElement
    
    // 移除之前的主题类
    root.classList.remove('light', 'dark')
    
    // 直接使用选择的主题
    root.classList.add(theme)
    setResolvedTheme(theme)
    
    // 保存到localStorage
    localStorage.setItem('theme', theme)
  }, [theme])

  // 移除系统主题监听相关代码

  return {
    theme,
    resolvedTheme,
    setTheme
  }
} 