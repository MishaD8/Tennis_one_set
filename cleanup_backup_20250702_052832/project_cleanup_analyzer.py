#!/usr/bin/env python3
"""
🧹 PROJECT CLEANUP ANALYZER
Анализирует файлы проекта и выявляет нужные/ненужные файлы
"""

import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime

class ProjectCleanupAnalyzer:
    """Анализатор для очистки проекта"""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.core_files = set()
        self.imported_files = set()
        self.backup_files = set()
        self.demo_files = set()
        self.duplicate_files = set()
        self.unused_files = set()
        
        # Критически важные файлы (никогда не удаляем)
        self.critical_files = {
            'config.json',
            'requirements.txt',
            'requirements_production.txt',
            'Dockerfile',
            'docker-compose.yml',
            '.env',
            '.gitignore',
            'README.md'
        }
        
        # Паттерны файлов для анализа
        self.file_patterns = {
            'backend': re.compile(r'.*backend.*\.py$'),
            'api': re.compile(r'.*api.*\.py$'),
            'predictor': re.compile(r'.*predictor.*\.py$'),
            'test': re.compile(r'.*test.*\.py$'),
            'backup': re.compile(r'.*backup.*|.*_backup_.*'),
            'demo': re.compile(r'.*demo.*|.*sample.*'),
            'config': re.compile(r'.*config.*\.(json|yml|yaml|ini)$'),
            'launcher': re.compile(r'.*start.*\.py$|.*launch.*\.py$'),
            'integration': re.compile(r'.*integration.*\.py$'),
            'odds': re.compile(r'.*odds.*\.py$'),
            'universal': re.compile(r'.*universal.*\.py$')
        }
    
    def analyze_imports(self, file_path: Path) -> Set[str]:
        """Анализирует импорты в Python файле"""
        imported_modules = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # AST анализ для точного определения импортов
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imported_modules.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imported_modules.add(node.module.split('.')[0])
            except:
                # Fallback к regex если AST не работает
                import_patterns = [
                    r'from\s+(\w+)',
                    r'import\s+(\w+)',
                ]
                
                for pattern in import_patterns:
                    matches = re.findall(pattern, content)
                    imported_modules.update(matches)
        
        except Exception as e:
            print(f"⚠️ Error analyzing {file_path}: {e}")
        
        return imported_modules
    
    def find_file_references(self, file_path: Path) -> Set[str]:
        """Ищет ссылки на другие файлы проекта"""
        references = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Ищем ссылки на .py файлы
            py_refs = re.findall(r'["\']([^"\']*\.py)["\']', content)
            references.update(py_refs)
            
            # Ищем ссылки на конфиг файлы
            config_refs = re.findall(r'["\']([^"\']*\.(json|yml|yaml|ini))["\']', content)
            references.update([ref[0] for ref in config_refs])
            
            # Ищем импорты локальных модулей
            local_imports = re.findall(r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import', content)
            references.update([f"{imp}.py" for imp in local_imports])
            
            # Ищем прямые импорты
            direct_imports = re.findall(r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            references.update([f"{imp}.py" for imp in direct_imports 
                              if not imp in ['os', 'sys', 'json', 'datetime', 'time', 'logging']])
        
        except Exception as e:
            print(f"⚠️ Error finding references in {file_path}: {e}")
        
        return references
    
    def categorize_files(self) -> Dict[str, List[Path]]:
        """Категоризирует файлы проекта"""
        categories = {
            'core_backend': [],
            'ml_models': [],
            'data_collectors': [],
            'api_integrations': [],
            'tests': [],
            'launchers': [],
            'configs': [],
            'backups': [],
            'demos': [],
            'duplicates': [],
            'universal_system': [],
            'other': []
        }
        
        # Получаем все Python и конфиг файлы
        files = list(self.project_dir.glob('*.py')) + list(self.project_dir.glob('*.json')) + list(self.project_dir.glob('*.yml'))
        
        for file_path in files:
            file_name = file_path.name.lower()
            
            # Бэкапы
            if 'backup' in file_name or '_backup_' in file_name:
                categories['backups'].append(file_path)
            
            # Демо файлы
            elif 'demo' in file_name or 'sample' in file_name or 'test' in file_name:
                if 'test' in file_name:
                    categories['tests'].append(file_path)
                else:
                    categories['demos'].append(file_path)
            
            # Универсальная система
            elif 'universal' in file_name:
                categories['universal_system'].append(file_path)
            
            # Backend файлы
            elif 'backend' in file_name or 'web_' in file_name:
                categories['core_backend'].append(file_path)
            
            # ML и предикторы
            elif any(word in file_name for word in ['predictor', 'prediction', 'model', 'ml']):
                categories['ml_models'].append(file_path)
            
            # Сборщики данных
            elif any(word in file_name for word in ['collector', 'data']):
                categories['data_collectors'].append(file_path)
            
            # API интеграции
            elif any(word in file_name for word in ['api', 'odds', 'integration']):
                categories['api_integrations'].append(file_path)
            
            # Лаунчеры
            elif any(word in file_name for word in ['start', 'launch', 'restart']):
                categories['launchers'].append(file_path)
            
            # Конфиги
            elif file_path.suffix in ['.json', '.yml', '.yaml']:
                categories['configs'].append(file_path)
            
            else:
                categories['other'].append(file_path)
        
        return categories
    
    def find_duplicates(self, categories: Dict[str, List[Path]]) -> List[Tuple[Path, Path, str]]:
        """Находит дублирующиеся файлы"""
        duplicates = []
        
        # Проверяем backend файлы
        backend_files = categories['core_backend']
        if len(backend_files) > 1:
            for i, file1 in enumerate(backend_files):
                for file2 in backend_files[i+1:]:
                    similarity = self.calculate_similarity(file1, file2)
                    if similarity > 0.8:
                        duplicates.append((file1, file2, f"Backend similarity: {similarity:.1%}"))
        
        # Проверяем launcher файлы
        launcher_files = categories['launchers']
        if len(launcher_files) > 1:
            for i, file1 in enumerate(launcher_files):
                for file2 in launcher_files[i+1:]:
                    similarity = self.calculate_similarity(file1, file2)
                    if similarity > 0.7:
                        duplicates.append((file1, file2, f"Launcher similarity: {similarity:.1%}"))
        
        return duplicates
    
    def calculate_similarity(self, file1: Path, file2: Path) -> float:
        """Вычисляет схожесть двух файлов"""
        try:
            with open(file1, 'r', encoding='utf-8', errors='ignore') as f:
                content1 = f.read()
            with open(file2, 'r', encoding='utf-8', errors='ignore') as f:
                content2 = f.read()
            
            # Простая схожесть по длине и общим строкам
            lines1 = set(content1.split('\n'))
            lines2 = set(content2.split('\n'))
            
            if len(lines1) == 0 or len(lines2) == 0:
                return 0.0
            
            common_lines = len(lines1.intersection(lines2))
            total_lines = len(lines1.union(lines2))
            
            return common_lines / total_lines if total_lines > 0 else 0.0
        
        except:
            return 0.0
    
    def analyze_usage(self, categories: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
        """Анализирует использование файлов"""
        all_files = []
        for file_list in categories.values():
            all_files.extend(file_list)
        
        file_references = {}
        
        # Находим все ссылки между файлами
        for file_path in all_files:
            if file_path.suffix == '.py':
                refs = self.find_file_references(file_path)
                file_references[file_path] = refs
        
        # Определяем какие файлы используются
        referenced_files = set()
        for refs in file_references.values():
            for ref in refs:
                ref_path = self.project_dir / ref
                if ref_path.exists():
                    referenced_files.add(ref_path)
        
        # Определяем неиспользуемые файлы
        unused = []
        for file_path in all_files:
            if (file_path.suffix == '.py' and 
                file_path not in referenced_files and 
                file_path.name not in self.critical_files):
                unused.append(file_path)
        
        return {
            'used': list(referenced_files),
            'unused': unused
        }
    
    def generate_recommendations(self) -> Dict[str, List[str]]:
        """Генерирует рекомендации по очистке"""
        
        categories = self.categorize_files()
        duplicates = self.find_duplicates(categories)
        usage = self.analyze_usage(categories)
        
        recommendations = {
            'safe_to_delete': [],
            'probably_delete': [],
            'keep_but_review': [],
            'definitely_keep': [],
            'actions_needed': []
        }
        
        # Безопасно удалить
        recommendations['safe_to_delete'].extend([
            str(f) for f in categories['backups']
        ])
        
        # Вероятно можно удалить
        recommendations['probably_delete'].extend([
            str(f) for f in categories['demos'] if 'test' not in f.name
        ])
        
        # Дубликаты
        for file1, file2, reason in duplicates:
            recommendations['probably_delete'].append(f"{file2} (duplicate of {file1.name})")
            recommendations['actions_needed'].append(f"Compare {file1.name} and {file2.name}: {reason}")
        
        # Неиспользуемые файлы
        for unused_file in usage['unused']:
            if unused_file.name not in self.critical_files:
                recommendations['probably_delete'].append(str(unused_file))
        
        # Определенно оставить
        recommendations['definitely_keep'].extend([
            str(f) for f in categories['configs'] if f.name in self.critical_files
        ])
        
        # Оставить активные backend
        active_backends = [f for f in categories['core_backend'] 
                          if 'backup' not in f.name and f in usage['used']]
        recommendations['definitely_keep'].extend([str(f) for f in active_backends])
        
        # Оставить ML модели
        recommendations['definitely_keep'].extend([
            str(f) for f in categories['ml_models'] if f in usage['used']
        ])
        
        return recommendations
    
    def create_cleanup_script(self, recommendations: Dict[str, List[str]]) -> str:
        """Создает скрипт для очистки"""
        
        script = f'''#!/usr/bin/env python3
"""
🧹 AUTOMATED PROJECT CLEANUP SCRIPT
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import os
import shutil
from pathlib import Path

def backup_before_cleanup():
    """Создает бэкап перед очисткой"""
    backup_dir = Path(f"cleanup_backup_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}")
    backup_dir.mkdir(exist_ok=True)
    return backup_dir

def safe_delete(file_path, backup_dir):
    """Безопасное удаление с бэкапом"""
    file_path = Path(file_path)
    if file_path.exists():
        # Копируем в бэкап
        shutil.copy2(file_path, backup_dir / file_path.name)
        # Удаляем оригинал
        file_path.unlink()
        print(f"✅ Deleted: {{file_path}}")
    else:
        print(f"⚠️ Not found: {{file_path}}")

def main():
    print("🧹 STARTING PROJECT CLEANUP")
    print("=" * 50)
    
    # Создаем бэкап
    backup_dir = backup_before_cleanup()
    print(f"💾 Backup directory: {{backup_dir}}")
    
    # Файлы для удаления
    files_to_delete = [
'''
        
        for file_path in recommendations['safe_to_delete'] + recommendations['probably_delete']:
            script += f'        "{file_path}",\n'
        
        script += '''    ]
    
    # Удаляем файлы
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            safe_delete(file_path, backup_dir)
            deleted_count += 1
        except Exception as e:
            print(f"❌ Error deleting {file_path}: {e}")
    
    print(f"\\n📊 CLEANUP SUMMARY:")
    print(f"🗑️ Files deleted: {deleted_count}")
    print(f"💾 Backup created: {backup_dir}")
    print(f"\\n✅ Cleanup completed!")

if __name__ == "__main__":
    import datetime
    main()
'''
        
        return script
    
    def run_analysis(self) -> Dict:
        """Запускает полный анализ"""
        print("🔍 ANALYZING PROJECT FILES...")
        print("=" * 50)
        
        categories = self.categorize_files()
        duplicates = self.find_duplicates(categories)
        usage = self.analyze_usage(categories)
        recommendations = self.generate_recommendations()
        
        return {
            'categories': categories,
            'duplicates': duplicates,
            'usage': usage,
            'recommendations': recommendations,
            'cleanup_script': self.create_cleanup_script(recommendations)
        }


def main():
    """Главная функция анализа"""
    
    print("🧹 PROJECT CLEANUP ANALYZER")
    print("=" * 50)
    print("🎯 Analyzing tennis prediction project files...")
    print("=" * 50)
    
    analyzer = ProjectCleanupAnalyzer()
    results = analyzer.run_analysis()
    
    # Выводим результаты
    print("\n📂 FILE CATEGORIES:")
    print("-" * 30)
    for category, files in results['categories'].items():
        if files:
            print(f"{category}: {len(files)} files")
            for file in files[:3]:  # Показываем первые 3
                print(f"  • {file.name}")
            if len(files) > 3:
                print(f"  ... and {len(files) - 3} more")
    
    print("\n🔄 DUPLICATE FILES:")
    print("-" * 30)
    if results['duplicates']:
        for file1, file2, reason in results['duplicates']:
            print(f"• {file1.name} ↔ {file2.name}")
            print(f"  Reason: {reason}")
    else:
        print("✅ No significant duplicates found")
    
    print("\n📊 USAGE ANALYSIS:")
    print("-" * 30)
    print(f"Used files: {len(results['usage']['used'])}")
    print(f"Unused files: {len(results['usage']['unused'])}")
    
    if results['usage']['unused']:
        print("Unused files:")
        for unused in results['usage']['unused'][:5]:
            print(f"  • {unused.name}")
    
    print("\n💡 CLEANUP RECOMMENDATIONS:")
    print("-" * 30)
    
    recs = results['recommendations']
    
    print(f"🗑️ Safe to delete ({len(recs['safe_to_delete'])} files):")
    for file in recs['safe_to_delete'][:5]:
        print(f"  • {Path(file).name}")
    
    print(f"\\n⚠️ Probably delete ({len(recs['probably_delete'])} files):")
    for file in recs['probably_delete'][:5]:
        print(f"  • {Path(file).name}")
    
    print(f"\\n✅ Definitely keep ({len(recs['definitely_keep'])} files):")
    for file in recs['definitely_keep'][:5]:
        print(f"  • {Path(file).name}")
    
    # Создаем cleanup script
    script_path = Path("cleanup_project.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(results['cleanup_script'])
    
    print(f"\\n🛠️ CLEANUP SCRIPT CREATED:")
    print(f"📄 File: {script_path}")
    print("\\n🚀 NEXT STEPS:")
    print("1. Review the recommendations above")
    print("2. Run: python cleanup_project.py")
    print("3. Check backup folder if you need to restore anything")
    
    # Сохраняем полный отчет
    report_path = Path("cleanup_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"PROJECT CLEANUP ANALYSIS REPORT\\n")
        f.write(f"Generated: {datetime.now()}\\n")
        f.write("=" * 50 + "\\n\\n")
        
        f.write("SAFE TO DELETE:\\n")
        for file in recs['safe_to_delete']:
            f.write(f"  {file}\\n")
        
        f.write("\\nPROBABLY DELETE:\\n")
        for file in recs['probably_delete']:
            f.write(f"  {file}\\n")
        
        f.write("\\nDEFINITELY KEEP:\\n")
        for file in recs['definitely_keep']:
            f.write(f"  {file}\\n")
    
    print(f"📋 Full report saved: {report_path}")
    
    return results

if __name__ == "__main__":
    main()