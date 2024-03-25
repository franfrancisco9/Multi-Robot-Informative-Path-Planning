import unittest
from radiation import RadiationField

class TestRadiationField(unittest.TestCase):
    def setUp(self):
        self.rf = RadiationField()

    def test_generate_sources(self):
        num_sources = 3
        workspace_size = (40, 40)
        intensity_range = (10000, 100000)
        sources = self.rf.generate_sources(num_sources, workspace_size, intensity_range)
        self.assertEqual(len(sources), num_sources)
        for source in sources:
            self.assertGreaterEqual(source[0], 0)
            self.assertLessEqual(source[0], workspace_size[0])
            self.assertGreaterEqual(source[1], 0)
            self.assertLessEqual(source[1], workspace_size[1])
            self.assertGreaterEqual(source[2], intensity_range[0])
            self.assertLessEqual(source[2], intensity_range[1])

    def test_update_source(self):
        source_index = 0
        new_x = 10
        new_y = 20
        new_A = 50000
        self.rf.update_source(source_index, new_x, new_y, new_A)
        sources = self.rf.get_sources_info()
        self.assertEqual(sources[source_index][0], new_x)
        self.assertEqual(sources[source_index][1], new_y)
        self.assertEqual(sources[source_index][2], new_A)

    def test_recalculate_ground_truth(self):
        self.rf.recalculate_ground_truth()
        g_truth = self.rf.ground_truth()
        self.assertEqual(g_truth.shape, self.rf.X.shape)

    def test_intensity(self):
        r = [10, 20]
        intensity = self.rf.intensity(r)
        self.assertIsInstance(intensity, float)

    def test_response(self):
        r = [10, 20]
        response = self.rf.response(r)
        self.assertIsInstance(response, float)

    def test_ground_truth(self):
        g_truth = self.rf.ground_truth()
        self.assertEqual(g_truth.shape, self.rf.X.shape)

    def test_simulate_measurements(self):
        waypoints = [[10, 20], [15, 25], [30, 35]]
        measurements = self.rf.simulate_measurements(waypoints)
        self.assertEqual(len(measurements), len(waypoints))
        for measurement in measurements:
            self.assertIsInstance(measurement, float)

    def test_predict_spatial_field(self):
        waypoints = [[10, 20], [15, 25], [30, 35]]
        measurements = [10000, 20000, 30000]
        Z_pred, std = self.rf.predict_spatial_field(waypoints, measurements)
        self.assertEqual(Z_pred.shape, self.rf.X.shape)
        self.assertEqual(std.shape[0], self.rf.X.shape[0]*self.rf.X.shape[1])

if __name__ == '__main__':
    unittest.main()