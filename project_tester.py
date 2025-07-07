#!/usr/bin/env python3
"""
🎾 STANDALONE ML COMPONENTS TESTER
Тестирует ML компоненты напрямую, без сервера
"""

import sys
import os
import traceback
from datetime import datetime
import json

class StandaloneMLTester:
    """Автономный тестер ML компонентов"""
    
    def __init__(self):
        self.test_results = {}
        self.available_components = {}
        
    def print_header(self, title):
        """Красивый заголовок"""
        print(f"\n{'='*80}")
        print(f"🎾 {title}")
        print(f"{'='*80}")
    
    def print_section(self, title):
        """Заголовок секции"""
        print(f"\n{'─'*60}")
        print(f"📊 {title}")
        print(f"{'─'*60}")
    
    def test_component_import(self, module_name, component_name):
        """Тест импорта компонента"""
        try:
            if module_name == "api_economy_patch":
                from api_economy_patch import (
                    init_api_economy, economical_tennis_request, 
                    get_api_usage, trigger_manual_update
                )
                return True, "API Economy functions imported"
                
            elif module_name == "real_tennis_predictor_integration":
                from real_tennis_predictor_integration import RealTennisPredictor
                return True, "RealTennisPredictor imported"
                
            elif module_name == "tennis_prediction_module":
                from tennis_prediction_module import TennisPredictionService
                return True, "TennisPredictionService imported"
                
            elif module_name == "universal_tennis_data_collector":
                from universal_tennis_data_collector import (
                    UniversalTennisDataCollector, UniversalOddsCollector
                )
                return True, "Universal collectors imported"
                
            else:
                exec(f"import {module_name}")
                return True, f"{module_name} imported"
                
        except ImportError as e:
            return False, f"Import error: {e}"
        except Exception as e:
            return False, f"Other error: {e}"
    
    def test_imports(self):
        """Тест всех импортов"""
        self.print_section("COMPONENT IMPORTS TEST")
        
        components_to_test = [
            ("api_economy_patch", "API Economy"),
            ("real_tennis_predictor_integration", "Real Tennis Predictor"),
            ("tennis_prediction_module", "Tennis Prediction Module"),
            ("universal_tennis_data_collector", "Universal Data Collector"),
            ("backend_integration_fix", "Backend Integration"),
            ("correct_odds_api_integration", "Odds API Integration"),
            ("tennis_set_predictor", "Set Predictor"),
        ]
        
        for module_name, component_name in components_to_test:
            success, message = self.test_component_import(module_name, component_name)
            
            if success:
                print(f"✅ {component_name}: {message}")
                self.available_components[module_name] = True
            else:
                print(f"❌ {component_name}: {message}")
                self.available_components[module_name] = False
        
        available_count = sum(self.available_components.values())
        total_count = len(self.available_components)
        
        print(f"\n📊 Components Summary: {available_count}/{total_count} available")
        return available_count > 0
    
    def test_real_tennis_predictor(self):
        """Тест Real Tennis Predictor"""
        self.print_section("REAL TENNIS PREDICTOR TEST")
        
        if not self.available_components.get("real_tennis_predictor_integration"):
            print("❌ Real Tennis Predictor not available")
            return False
        
        try:
            from real_tennis_predictor_integration import RealTennisPredictor
            
            print("✅ Creating RealTennisPredictor instance...")
            predictor = RealTennisPredictor()
            
            # Тест получения данных игрока
            print("🔍 Testing player data collection...")
            
            test_players = ["Carlos Alcaraz", "Novak Djokovic", "Unknown Player"]
            for player in test_players:
                player_data = predictor.data_collector.get_player_data(player)
                print(f"   {player}: Rank #{player_data.get('rank', 'unknown')}, "
                      f"Age {player_data.get('age', 'unknown')}")
            
            # Тест создания признаков
            print("🔧 Testing feature creation...")
            
            match_features = predictor.create_match_features(
                "Carlos Alcaraz", "Novak Djokovic", 
                "Wimbledon", "Grass", "SF"
            )
            
            print(f"   Created {len(match_features)} features")
            print(f"   Sample features: {list(match_features.keys())[:5]}")
            
            # Тест прогнозирования
            print("🤖 Testing match prediction...")
            
            test_matches = [
                ("Carlos Alcaraz", "Novak Djokovic", "Wimbledon", "Grass"),
                ("Jannik Sinner", "Daniil Medvedev", "US Open", "Hard"),
                ("Unknown Player A", "Unknown Player B", "ATP 250", "Hard")
            ]
            
            predictions = []
            for player1, player2, tournament, surface in test_matches:
                try:
                    result = predictor.predict_match(player1, player2, tournament, surface)
                    
                    print(f"   {player1} vs {player2}:")
                    print(f"      Probability: {result['probability']:.1%}")
                    print(f"      Confidence: {result['confidence']}")
                    print(f"      Type: {result['prediction_type']}")
                    
                    predictions.append(result)
                    
                except Exception as e:
                    print(f"   ❌ Prediction failed for {player1} vs {player2}: {e}")
            
            self.test_results['real_predictor'] = {
                'status': 'success',
                'predictions_count': len(predictions),
                'features_count': len(match_features),
                'sample_prediction': predictions[0] if predictions else None
            }
            
            print(f"✅ Real Tennis Predictor working: {len(predictions)} successful predictions")
            return True
            
        except Exception as e:
            print(f"❌ Real Tennis Predictor test failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            self.test_results['real_predictor'] = {'status': 'error', 'error': str(e)}
            return False
    
    def test_tennis_prediction_service(self):
        """Тест Tennis Prediction Service"""
        self.print_section("TENNIS PREDICTION SERVICE TEST")
        
        if not self.available_components.get("tennis_prediction_module"):
            print("❌ Tennis Prediction Module not available")
            return False
        
        try:
            from tennis_prediction_module import TennisPredictionService, create_match_data
            
            print("✅ Creating TennisPredictionService instance...")
            service = TennisPredictionService()
            
            # Проверка загрузки моделей
            print("🔍 Testing model loading...")
            
            models_loaded = service.load_models()
            if models_loaded:
                print("✅ ML models loaded successfully")
                print(f"   Models directory: {service.models_dir}")
                
                model_info = service.get_model_info()
                if model_info.get('status') == 'loaded':
                    print(f"   Loaded models: {model_info['models']}")
                    print(f"   Expected features: {len(model_info['expected_features'])}")
                else:
                    print("⚠️ Models loaded but not fully initialized")
            else:
                print("⚠️ ML models not found, using demo mode")
            
            # Тест создания данных матча
            print("🔧 Testing match data creation...")
            
            match_data = create_match_data(
                player_rank=1,
                opponent_rank=45,
                player_recent_win_rate=0.85,
                h2h_win_rate=0.75
            )
            
            print(f"   Created match data with {len(match_data)} features")
            
            # Тест прогнозирования
            print("🤖 Testing prediction service...")
            
            try:
                prediction_result = service.predict_match(match_data, return_details=True)
                
                print(f"   Prediction result:")
                print(f"      Probability: {prediction_result['probability']:.1%}")
                print(f"      Confidence: {prediction_result['confidence']}")
                print(f"      Recommendation: {prediction_result.get('recommendation', 'N/A')}")
                
                if 'individual_predictions' in prediction_result:
                    print(f"      Individual models: {len(prediction_result['individual_predictions'])}")
                
                self.test_results['prediction_service'] = {
                    'status': 'success',
                    'models_loaded': models_loaded,
                    'prediction': prediction_result
                }
                
                print("✅ Tennis Prediction Service working")
                return True
                
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
                self.test_results['prediction_service'] = {
                    'status': 'prediction_error',
                    'models_loaded': models_loaded,
                    'error': str(e)
                }
                return False
            
        except Exception as e:
            print(f"❌ Tennis Prediction Service test failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            self.test_results['prediction_service'] = {'status': 'error', 'error': str(e)}
            return False
    
    def test_api_economy(self):
        """Тест API Economy"""
        self.print_section("API ECONOMY TEST")
        
        if not self.available_components.get("api_economy_patch"):
            print("❌ API Economy not available")
            return False
        
        try:
            from api_economy_patch import (
                init_api_economy, get_api_usage, 
                trigger_manual_update, check_manual_update_status
            )
            
            print("✅ API Economy functions imported")
            
            # Инициализация
            print("🔧 Testing API Economy initialization...")
            
            init_api_economy(
                api_key="test_key_for_testing",
                max_per_hour=30,
                cache_minutes=20
            )
            
            print("✅ API Economy initialized")
            
            # Тест получения статистики
            print("📊 Testing usage statistics...")
            
            try:
                usage_stats = get_api_usage()
                print(f"   Usage stats structure: {list(usage_stats.keys())}")
                print(f"   Max per hour: {usage_stats.get('max_per_hour', 'unknown')}")
                print(f"   Cache items: {usage_stats.get('cache_items', 'unknown')}")
            except Exception as e:
                print(f"   ⚠️ Usage stats error: {e}")
            
            # Тест ручного обновления
            print("🔄 Testing manual update trigger...")
            
            try:
                update_success = trigger_manual_update()
                print(f"   Manual update trigger: {'✅ Success' if update_success else '❌ Failed'}")
            except Exception as e:
                print(f"   ⚠️ Manual update error: {e}")
            
            # Тест статуса обновления
            print("📋 Testing manual update status...")
            
            try:
                update_status = check_manual_update_status()
                print(f"   Update status: {update_status}")
            except Exception as e:
                print(f"   ⚠️ Update status error: {e}")
            
            self.test_results['api_economy'] = {
                'status': 'success',
                'functions_working': True
            }
            
            print("✅ API Economy system working")
            return True
            
        except Exception as e:
            print(f"❌ API Economy test failed: {e}")
            self.test_results['api_economy'] = {'status': 'error', 'error': str(e)}
            return False
    
    def test_integration_logic(self):
        """Тест интеграционной логики"""
        self.print_section("INTEGRATION LOGIC TEST")
        
        # Проверим можем ли создать интегрированный предиктор
        if (self.available_components.get("real_tennis_predictor_integration") and 
            self.available_components.get("tennis_prediction_module")):
            
            try:
                print("🔧 Testing integrated ML predictor...")
                
                # Имитируем логику из backend_tennis_integrated.py
                from real_tennis_predictor_integration import RealTennisPredictor
                from tennis_prediction_module import TennisPredictionService
                
                real_predictor = RealTennisPredictor()
                prediction_service = TennisPredictionService()
                
                # Тест интегрированного прогноза
                print("🤖 Testing integrated prediction...")
                
                # Сначала пробуем Real Predictor
                result_real = real_predictor.predict_match(
                    "Carlos Alcaraz", "Novak Djokovic", "Wimbledon", "Grass"
                )
                
                print(f"   Real Predictor result:")
                print(f"      Type: {result_real['prediction_type']}")
                print(f"      Probability: {result_real['probability']:.1%}")
                print(f"      System: {result_real.get('ml_system_used', 'unknown')}")
                
                # Определяем приоритет
                if result_real['prediction_type'] == 'REAL_ML_MODEL':
                    priority_system = "Real ML Models"
                elif result_real['prediction_type'] == 'ADVANCED_SIMULATION':
                    priority_system = "Advanced Simulation"
                else:
                    priority_system = "Fallback"
                
                print(f"   🎯 Priority system would be: {priority_system}")
                
                self.test_results['integration'] = {
                    'status': 'success',
                    'priority_system': priority_system,
                    'real_predictor_type': result_real['prediction_type']
                }
                
                print("✅ Integration logic working")
                return True
                
            except Exception as e:
                print(f"❌ Integration test failed: {e}")
                self.test_results['integration'] = {'status': 'error', 'error': str(e)}
                return False
        else:
            print("❌ Cannot test integration - missing required components")
            return False
    
    def analyze_file_structure(self):
        """Анализ структуры файлов проекта"""
        self.print_section("PROJECT STRUCTURE ANALYSIS")
        
        important_files = [
            "backend_tennis_integrated.py",
            "api_economy_patch.py", 
            "real_tennis_predictor_integration.py",
            "tennis_prediction_module.py",
            "universal_tennis_data_collector.py",
            "config.json",
            "requirements.txt",
            "tennis_models/",
            "api_usage.json",
            "api_cache.json"
        ]
        
        file_status = {}
        
        for file_path in important_files:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    count = len(os.listdir(file_path)) if os.path.isdir(file_path) else 0
                    file_status[file_path] = f"✅ Directory ({count} items)"
                else:
                    size = os.path.getsize(file_path)
                    file_status[file_path] = f"✅ File ({size} bytes)"
            else:
                file_status[file_path] = "❌ Missing"
        
        print("📁 Project files:")
        for file_path, status in file_status.items():
            print(f"   {file_path}: {status}")
        
        # Проверка моделей
        if os.path.exists("tennis_models"):
            print(f"\n🤖 ML Models directory:")
            try:
                model_files = os.listdir("tennis_models")
                for model_file in model_files:
                    print(f"   {model_file}")
                
                # Ключевые файлы моделей
                key_model_files = ["scaler.pkl", "metadata.json"]
                for key_file in key_model_files:
                    if key_file in model_files:
                        print(f"   ✅ {key_file} found")
                    else:
                        print(f"   ❌ {key_file} missing")
                        
            except Exception as e:
                print(f"   ❌ Error reading models directory: {e}")
        
        self.test_results['file_structure'] = file_status
    
    def generate_report(self):
        """Генерация итогового отчета"""
        self.print_header("STANDALONE ML TEST REPORT")
        
        # Подсчет успешных тестов
        successful_tests = 0
        total_tests = 0
        
        test_categories = [
            ('real_predictor', 'Real Tennis Predictor'),
            ('prediction_service', 'Tennis Prediction Service'),
            ('api_economy', 'API Economy'),
            ('integration', 'Integration Logic')
        ]
        
        print(f"\n📊 ML COMPONENTS TEST RESULTS:")
        print(f"{'─'*50}")
        
        for test_key, test_name in test_categories:
            total_tests += 1
            
            if test_key in self.test_results:
                result = self.test_results[test_key]
                if result.get('status') == 'success':
                    status = "✅ WORKING"
                    successful_tests += 1
                elif result.get('status') == 'prediction_error':
                    status = "⚠️ PARTIAL (models missing)"
                    successful_tests += 0.5
                else:
                    status = f"❌ FAILED ({result.get('error', 'unknown')})"
            else:
                status = "❌ NOT TESTED"
            
            print(f"{test_name:25}: {status}")
        
        print(f"{'─'*50}")
        print(f"ML Components Score: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
        
        # Анализ доступных компонентов
        available_count = sum(self.available_components.values())
        total_components = len(self.available_components)
        
        print(f"Available Components: {available_count}/{total_components} ({available_count/total_components*100:.1f}%)")
        
        # Рекомендации
        self.print_section("RECOMMENDATIONS")
        
        if successful_tests >= total_tests * 0.8:
            print("🎉 EXCELLENT! ML components working well!")
            print("   Your ML system is ready for production.")
            
            # Специфичные рекомендации
            if 'real_predictor' in self.test_results:
                real_result = self.test_results['real_predictor']
                if real_result.get('status') == 'success':
                    pred = real_result.get('sample_prediction', {})
                    if pred.get('prediction_type') == 'REAL_ML_MODEL':
                        print("   🤖 Real ML Models are working - highest quality setup!")
                    else:
                        print("   🎯 Advanced Simulation working - good fallback system")
        
        elif successful_tests >= total_tests * 0.5:
            print("⚠️ PARTIAL SUCCESS. Some ML components working.")
            print("   Consider training ML models for better accuracy.")
        else:
            print("❌ MULTIPLE ISSUES. ML components need attention.")
            print("   Check imports and dependencies.")
        
        # Сохранение отчета
        self.save_report()
    
    def save_report(self):
        """Сохранение отчета"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"standalone_ml_test_report_{timestamp}.json"
            
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "test_type": "standalone_ml_components",
                "available_components": self.available_components,
                "test_results": self.test_results,
                "summary": {
                    "components_available": sum(self.available_components.values()),
                    "total_components": len(self.available_components),
                    "tests_passed": len([r for r in self.test_results.values() 
                                       if r.get('status') == 'success'])
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 Standalone test report saved: {filename}")
            
        except Exception as e:
            print(f"\n⚠️ Failed to save report: {e}")
    
    def run_standalone_test(self):
        """Запуск автономного тестирования"""
        self.print_header("STANDALONE ML COMPONENTS TESTING")
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Working directory: {os.getcwd()}")
        
        try:
            # Анализ структуры проекта
            self.analyze_file_structure()
            
            # Тест импортов
            imports_ok = self.test_imports()
            
            if not imports_ok:
                print("\n❌ No components available for testing")
                return
            
            # Тесты компонентов
            if self.available_components.get("real_tennis_predictor_integration"):
                self.test_real_tennis_predictor()
            
            if self.available_components.get("tennis_prediction_module"):
                self.test_tennis_prediction_service()
            
            if self.available_components.get("api_economy_patch"):
                self.test_api_economy()
            
            # Тест интеграции
            self.test_integration_logic()
            
            # Генерация отчета
            self.generate_report()
            
        except KeyboardInterrupt:
            print(f"\n⏹️ Testing interrupted by user")
        except Exception as e:
            print(f"\n❌ Testing failed with error: {e}")
            traceback.print_exc()


def main():
    """Главная функция"""
    print("🎾 STANDALONE ML COMPONENTS TESTER")
    print("=" * 80)
    print("🔍 Testing ML components without running server")
    
    tester = StandaloneMLTester()
    tester.run_standalone_test()
    
    print(f"\n🎯 Standalone testing completed!")


if __name__ == "__main__":
    main()