"""Validate experiment arguments and command forwarding without using a lab GPU."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import run_lab
from run_lab import parse_args, run_name, command, MODELS


class LauncherTests(unittest.TestCase):
    def test_required_arguments_and_invalid_numbers(self):
        cases = [[], ['3090', MODELS[1], '50', '.0001'],
                 ['3090', MODELS[1], '-1', '.0001', '250'],
                 ['3090', MODELS[1], '50', 'nan', '250'],
                 ['3090', MODELS[1], '50', '0', '250'],
                 ['3090', MODELS[1], '50', '.0001', '0']]
        for argv in cases:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_comments_and_equivalent_rates(self):
        a = parse_args(['3090', MODELS[1], '50', '.0001', '250', 'Changed learning rate'])
        b = parse_args(['3090', MODELS[1], '50', '1e-4', '250', 'changed-learning-rate'])
        self.assertEqual(run_name(a), run_name(b))
        self.assertTrue(run_name(a).endswith('-changed-learning-rate'))
        a.comments = ''
        self.assertTrue(run_name(a).endswith('-e250'))

    def test_both_models_and_environments_forward_exact_settings(self):
        for env in ('3090', '4070'):
            for model in MODELS:
                args = parse_args([env, model, '78', '.002', '13', '--resume'])
                cmd = command(args, Path('/project'), Path('/python'), Path('/data'))
                for flag, value in [('--model', model), ('--seed', '78'),
                                    ('--learning-rate', '0.002'), ('--epochs', '13')]:
                    self.assertEqual(cmd[cmd.index(flag)+1], value)
                self.assertEqual(cmd[-1], '--resume')
                self.assertIn(f'{env}-{model}-s78-lr0.002-e13', cmd[cmd.index('--output')+1])

    def test_logging_resume_and_exit_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python = root / 'fake-python'
            python.write_text('#!/bin/sh\necho training-output\nexit 7\n')
            python.chmod(0o700)
            dataset = root / 'data.pickle'
            dataset.touch()
            argv = ['run_lab.py', '3090', MODELS[1], '50', '.0001', '250', 'test run']
            args = parse_args(argv[1:])
            with patch.object(run_lab, '__file__', str(root / 'run_lab.py')), \
                 patch('run_lab.socket.gethostname', return_value='cs23002lv'), \
                 patch.dict('os.environ', {'COSC440_PYTHON': str(python), 'COSC440_DATASET': str(dataset)}), \
                 patch('sys.argv', argv), patch('sys.stdout', new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as result:
                    run_lab.main()
                self.assertEqual(result.exception.code, 7)
                log = root / 'logs' / (run_name(args) + '.log')
                self.assertIn('training-output', log.read_text())
                self.assertIn('Process exit status: 7', log.read_text())
                before = log.read_text()
                with self.assertRaises(FileExistsError):
                    run_lab.main()
                self.assertEqual(log.read_text(), before)
                output = root / 'runs' / run_name(args)
                output.mkdir(parents=True)
                (output / 'config.json').write_text('{}')
                with patch('sys.argv', argv + ['--resume']), self.assertRaises(SystemExit):
                    run_lab.main()
                self.assertEqual(log.read_text().count('training-output'), 2)


if __name__ == '__main__':
    unittest.main()
