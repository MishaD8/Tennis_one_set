"""
Unit tests for UniversalTennisDataCollector
"""
import pytest
from datetime import datetime, timedelta, date
from unittest.mock import Mock, patch
import sys
import os

# Import the module under test
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from universal_tennis_data_collector import UniversalTennisDataCollector


class TestUniversalTennisDataCollector:
    """Test suite for UniversalTennisDataCollector class"""
    
    @pytest.fixture
    def collector(self):
        """Create UniversalTennisDataCollector instance"""
        return UniversalTennisDataCollector()
    
    @pytest.fixture
    def mock_current_date(self):
        """Mock current date for consistent testing"""
        return datetime(2025, 7, 15)  # Mid-year date for testing
    
    def test_init_creates_tournament_calendar(self, collector):
        """Test that initialization creates tournament calendar"""
        assert hasattr(collector, 'tournament_calendar')
        assert isinstance(collector.tournament_calendar, dict)
        assert len(collector.tournament_calendar) > 0
        
        # Check that current_date is set
        assert hasattr(collector, 'current_date')
        assert isinstance(collector.current_date, datetime)
    
    def test_tournament_calendar_structure(self, collector):
        """Test tournament calendar has proper structure"""
        calendar = collector.tournament_calendar
        
        # Check a few specific tournaments exist
        expected_tournaments = [
            "2025-01-12",  # Australian Open
            "2025-05-19",  # French Open
            "2025-06-30",  # Wimbledon
            "2025-08-25"   # US Open
        ]
        
        for date_str in expected_tournaments:
            assert date_str in calendar
            tournament = calendar[date_str]
            
            # Check required fields
            required_fields = ["name", "location", "surface", "level", "status"]
            for field in required_fields:
                assert field in tournament
                assert isinstance(tournament[field], str)
    
    def test_tournament_calendar_grand_slams(self, collector):
        """Test that all Grand Slams are included in calendar"""
        calendar = collector.tournament_calendar
        grand_slams = [
            ("Australian Open", "Hard"),
            ("French Open", "Clay"),
            ("Wimbledon", "Grass"),
            ("US Open", "Hard")
        ]
        
        found_slams = []
        for tournament in calendar.values():
            if tournament["level"] == "Grand Slam":
                found_slams.append((tournament["name"], tournament["surface"]))
        
        assert len(found_slams) == 4
        for slam in grand_slams:
            assert slam in found_slams
    
    def test_tournament_calendar_surface_variety(self, collector):
        """Test that calendar includes all surface types"""
        calendar = collector.tournament_calendar
        surfaces = set()
        
        for tournament in calendar.values():
            surfaces.add(tournament["surface"])
        
        expected_surfaces = {"Hard", "Clay", "Grass"}
        assert expected_surfaces.issubset(surfaces)
    
    @patch('universal_tennis_data_collector.datetime')
    def test_get_current_active_tournaments_specific_date(self, mock_datetime, collector):
        """Test getting active tournaments for a specific date"""
        # Mock current date to be during Wimbledon
        test_date = datetime(2025, 7, 5)  # During Wimbledon period
        mock_datetime.now.return_value = test_date
        
        # Re-initialize collector with mocked date
        collector.current_date = test_date
        
        active_tournaments = collector.get_current_active_tournaments()
        
        assert isinstance(active_tournaments, list)
        # Should find at least one tournament (this depends on implementation)
        # The exact assertion depends on the date logic in the implementation
    
    @patch('universal_tennis_data_collector.datetime')
    def test_get_current_active_tournaments_no_active(self, mock_datetime, collector):
        """Test getting active tournaments when none are active"""
        # Mock date when no tournaments are running
        test_date = datetime(2025, 12, 31)  # End of year
        mock_datetime.now.return_value = test_date
        
        collector.current_date = test_date
        
        active_tournaments = collector.get_current_active_tournaments()
        
        assert isinstance(active_tournaments, list)
        # Could be empty or have fallback tournaments
    
    def test_tournament_calendar_date_format(self, collector):
        """Test that tournament calendar uses correct date format"""
        calendar = collector.tournament_calendar
        
        for date_str in calendar.keys():
            # Should be in YYYY-MM-DD format
            assert len(date_str) == 10
            assert date_str.count('-') == 2
            
            # Should be parseable as date
            try:
                year, month, day = date_str.split('-')
                datetime(int(year), int(month), int(day))
            except ValueError:
                pytest.fail(f"Invalid date format: {date_str}")
    
    def test_tournament_status_categories(self, collector):
        """Test that tournaments have appropriate status categories"""
        calendar = collector.tournament_calendar
        valid_statuses = {"major", "masters", "atp", "wta", "team", "exhibition", "finals"}
        
        found_statuses = set()
        for tournament in calendar.values():
            status = tournament["status"]
            found_statuses.add(status)
            assert status in valid_statuses
        
        # Should have multiple status types
        assert len(found_statuses) > 1
    
    def test_tournament_level_categories(self, collector):
        """Test that tournaments have appropriate level categories"""
        calendar = collector.tournament_calendar
        valid_levels = {
            "Grand Slam", "ATP 1000", "ATP 500", "ATP 250", 
            "WTA 1000", "WTA 500", "WTA 250", "Team Event", 
            "Exhibition", "Finals"
        }
        
        found_levels = set()
        for tournament in calendar.values():
            level = tournament["level"]
            found_levels.add(level)
            # Level should be one of the valid categories
            assert any(valid_level in level for valid_level in valid_levels)
        
        # Should have multiple level types
        assert len(found_levels) > 1
    
    def test_tournament_calendar_chronological_order(self, collector):
        """Test that tournaments are in chronological order within the year"""
        calendar = collector.tournament_calendar
        dates = list(calendar.keys())
        
        # Convert to datetime objects for comparison
        date_objects = []
        for date_str in dates:
            year, month, day = date_str.split('-')
            date_objects.append(date(int(year), int(month), int(day)))
        
        # Dates should be in ascending order
        for i in range(1, len(date_objects)):
            assert date_objects[i] >= date_objects[i-1]
    
    def test_tournament_calendar_year_coverage(self, collector):
        """Test that calendar covers the entire year"""
        calendar = collector.tournament_calendar
        months_covered = set()
        
        for date_str in calendar.keys():
            month = int(date_str.split('-')[1])
            months_covered.add(month)
        
        # Should cover most months of the year
        assert len(months_covered) >= 9  # At least 9 months covered
    
    @pytest.mark.parametrize("test_date,expected_behavior", [
        (datetime(2025, 1, 15), "during_australian_open"),
        (datetime(2025, 6, 5), "before_wimbledon"),
        (datetime(2025, 7, 10), "during_wimbledon"),
        (datetime(2025, 9, 1), "after_us_open")
    ])
    def test_get_current_active_tournaments_various_dates(self, collector, test_date, expected_behavior):
        """Test active tournaments for various dates throughout the year"""
        with patch('universal_tennis_data_collector.datetime') as mock_datetime:
            mock_datetime.now.return_value = test_date
            collector.current_date = test_date
            
            active_tournaments = collector.get_current_active_tournaments()
            
            assert isinstance(active_tournaments, list)
            # The specific assertions would depend on the implementation logic
            # This test ensures the method doesn't crash for various dates
    
    def test_tournament_data_integrity(self, collector):
        """Test that all tournament data is complete and valid"""
        calendar = collector.tournament_calendar
        
        for date_str, tournament in calendar.items():
            # Name should not be empty
            assert tournament["name"].strip() != ""
            
            # Location should not be empty
            assert tournament["location"].strip() != ""
            
            # Surface should be valid
            assert tournament["surface"] in ["Hard", "Clay", "Grass", "Various"]
            
            # Level should contain relevant keywords
            level_keywords = ["Grand Slam", "ATP", "WTA", "Masters", "Open", "Cup", "Finals", "Event", "Exhibition"]
            assert any(keyword in tournament["level"] for keyword in level_keywords)
    
    def test_major_tournaments_included(self, collector):
        """Test that major tennis events are included"""
        calendar = collector.tournament_calendar
        tournament_names = [t["name"] for t in calendar.values()]
        
        major_tournaments = [
            "Australian Open",
            "French Open", 
            "Wimbledon",
            "US Open",
            "ATP Finals",
            "WTA Finals"
        ]
        
        for major in major_tournaments:
            assert any(major in name for name in tournament_names)
    
    def test_surface_distribution(self, collector):
        """Test that there's a good distribution of surface types"""
        calendar = collector.tournament_calendar
        surface_count = {"Hard": 0, "Clay": 0, "Grass": 0, "Various": 0}
        
        for tournament in calendar.values():
            surface = tournament["surface"]
            if surface in surface_count:
                surface_count[surface] += 1
        
        # Hard courts should be most common
        assert surface_count["Hard"] > surface_count["Clay"]
        assert surface_count["Hard"] > surface_count["Grass"]
        
        # Should have representation of all major surfaces
        assert surface_count["Hard"] > 0
        assert surface_count["Clay"] > 0
        assert surface_count["Grass"] > 0


class TestUniversalTennisDataCollectorEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_tournament_calendar_handling(self):
        """Test handling when tournament calendar is empty"""
        collector = UniversalTennisDataCollector()
        
        # Mock empty calendar
        collector.tournament_calendar = {}
        
        active_tournaments = collector.get_current_active_tournaments()
        
        assert isinstance(active_tournaments, list)
        assert len(active_tournaments) == 0
    
    @patch('universal_tennis_data_collector.datetime')
    def test_leap_year_handling(self, mock_datetime):
        """Test handling of leap year dates"""
        # Test with leap year date
        leap_year_date = datetime(2024, 2, 29)
        mock_datetime.now.return_value = leap_year_date
        
        collector = UniversalTennisDataCollector()
        collector.current_date = leap_year_date
        
        # Should not crash
        active_tournaments = collector.get_current_active_tournaments()
        assert isinstance(active_tournaments, list)
    
    @patch('universal_tennis_data_collector.datetime')
    def test_year_boundary_handling(self, mock_datetime):
        """Test handling at year boundaries"""
        # Test at end of year
        end_of_year = datetime(2025, 12, 31, 23, 59, 59)
        mock_datetime.now.return_value = end_of_year
        
        collector = UniversalTennisDataCollector()
        collector.current_date = end_of_year
        
        active_tournaments = collector.get_current_active_tournaments()
        assert isinstance(active_tournaments, list)
        
        # Test at beginning of year
        start_of_year = datetime(2025, 1, 1, 0, 0, 1)
        mock_datetime.now.return_value = start_of_year
        
        collector.current_date = start_of_year
        active_tournaments = collector.get_current_active_tournaments()
        assert isinstance(active_tournaments, list)
    
    def test_malformed_date_handling(self):
        """Test handling of malformed dates in calendar"""
        collector = UniversalTennisDataCollector()
        
        # Add malformed date to calendar
        collector.tournament_calendar["invalid-date"] = {
            "name": "Test Tournament",
            "location": "Test Location",
            "surface": "Hard",
            "level": "Test Level",
            "status": "test"
        }
        
        # Should handle gracefully without crashing
        try:
            active_tournaments = collector.get_current_active_tournaments()
            assert isinstance(active_tournaments, list)
        except Exception as e:
            # If it raises an exception, it should be a specific, expected one
            assert isinstance(e, (ValueError, TypeError))


@pytest.mark.integration
class TestUniversalTennisDataCollectorIntegration:
    """Integration tests for the data collector"""
    
    def test_real_world_date_scenarios(self):
        """Test with real-world date scenarios"""
        collector = UniversalTennisDataCollector()
        
        # Test with current actual date
        real_active_tournaments = collector.get_current_active_tournaments()
        
        assert isinstance(real_active_tournaments, list)
        # The exact content depends on the current date
        
        # Each tournament should have proper structure
        for tournament in real_active_tournaments:
            assert isinstance(tournament, dict)
            required_fields = ["name", "location", "surface", "level", "status"]
            for field in required_fields:
                assert field in tournament
    
    def test_tournament_calendar_consistency(self):
        """Test that tournament calendar is consistent across instances"""
        collector1 = UniversalTennisDataCollector()
        collector2 = UniversalTennisDataCollector()
        
        # Calendars should be identical
        assert collector1.tournament_calendar == collector2.tournament_calendar
    
    def test_performance_with_large_calendar(self):
        """Test performance with large tournament calendar"""
        collector = UniversalTennisDataCollector()
        
        # Add many tournaments to test performance
        large_calendar = collector.tournament_calendar.copy()
        for i in range(1000):
            date_str = f"2025-01-{(i % 28) + 1:02d}"
            large_calendar[f"{date_str}-{i}"] = {
                "name": f"Test Tournament {i}",
                "location": f"Location {i}",
                "surface": "Hard",
                "level": "ATP 250",
                "status": "atp"
            }
        
        collector.tournament_calendar = large_calendar
        
        # Should still work efficiently
        import time
        start_time = time.time()
        active_tournaments = collector.get_current_active_tournaments()
        end_time = time.time()
        
        assert isinstance(active_tournaments, list)
        # Should complete within reasonable time (adjust threshold as needed)
        assert (end_time - start_time) < 1.0  # Less than 1 second