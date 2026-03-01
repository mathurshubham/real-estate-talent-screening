# Goal Description
Complete the TypeScript migration for the entire monorepo, ensuring there are absolutely no `.js` or `.jsx` files remaining. This includes migrating the root helper scripts and all frontend configuration files to TypeScript.

## Proposed Changes

### Root Directory
#### [MODIFY] package.json(file:///home/shubham/Documents/projects/sandbox/package.json)
- Add `tsx` to `devDependencies` to allow running TypeScript scripts directly without a build step.

#### [NEW] check_models.ts(file:///home/shubham/Documents/projects/sandbox/check_models.ts)
- Rename and migrate `check_models.js` to `check_models.ts`.
- Add proper types for the Gemini API interactions.

#### [DELETE] check_models.js(file:///home/shubham/Documents/projects/sandbox/check_models.js)

### Frontend Directory
#### [NEW] vite.config.ts(file:///home/shubham/Documents/projects/sandbox/frontend/vite.config.ts)
- Rename and migrate `vite.config.js` to `vite.config.ts`.

#### [NEW] tailwind.config.ts(file:///home/shubham/Documents/projects/sandbox/frontend/tailwind.config.ts)
- Rename and migrate `tailwind.config.js` to `tailwind.config.ts`.
- Update `content` paths to look for `.ts` and `.tsx` files instead of `.js` and `.jsx`.

#### [NEW] postcss.config.ts(file:///home/shubham/Documents/projects/sandbox/frontend/postcss.config.ts)
- Rename and migrate `postcss.config.js` to `postcss.config.ts`.

#### [NEW] eslint.config.ts(file:///home/shubham/Documents/projects/sandbox/frontend/eslint.config.ts)
- Rename and migrate `eslint.config.js` to `eslint.config.ts`.
- Update the `files` pattern to target `.ts` and `.tsx`.

#### [DELETE] All .js files in frontend/ root
- Remove legacy configuration files.

## Verification Plan

### Automated Tests
1. **Frontend Build**: Run `npm run build` in the `frontend` directory. Vite and TypeScript should correctly pick up the new `.ts` configuration files.
2. **Script Execution**: Run `npx tsx check_models.ts` in the root to ensure the script still functions as expected.
3. **Linting**: Run `npm run lint` in the `frontend` to verify that the new `eslint.config.ts` is correctly processed by ESLint.

### Manual Verification
- Confirm that no `.js` or `.jsx` files remain in the project using `find . -name "*.js" -o -name "*.jsx"`.
