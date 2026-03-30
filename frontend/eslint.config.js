import js from '@eslint/js';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import tseslint from 'typescript-eslint';

export default [
  {
    ignores: ['dist/**', 'node_modules/**'],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      // TypeScript 已负责全局标识符解析；在浏览器环境下启用 no-undef 会对 window/document 等产生大量误报。
      'no-undef': 'off',
      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      '@typescript-eslint/no-unused-expressions': [
        'error',
        { allowShortCircuit: true, allowTernary: true, allowTaggedTemplates: true },
      ],
      // 现有代码包含较多 any（例如 SSE/后端返回结构未完全类型化）；在不改变业务逻辑的前提下先允许 any，后续可逐步收紧。
      '@typescript-eslint/no-explicit-any': 'off',
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': [
        'error',
        { allowConstantExport: true },
      ],
    },
  },
];
