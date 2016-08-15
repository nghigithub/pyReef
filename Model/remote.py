import os

from ipyparallel import Client


def relog():
    ''' For debugging, redirect the individual node stdout/stderr to a file '''
    import os
    pid = os.getpid()
    logfile = open('/tmp/model-%s.txt' % pid, 'w')
    logfile.write('--- I am PID %s\n' % pid)

    import sys
    sys.stdout = logfile
    sys.stderr = logfile


class RemoteModel(object):
    """
    Wrapper to allow Model to run on an MPI cluster while hiding most of the
    MPI details.

    The public interface is identical to Model, so you should be able to
    substitute one for the other as desired..

    We use MPI in a master-slave architecture. This object runs on the master
    and handles all communication with the slaves. The slaves run the native
    Model object. In this way, the calling code does not need to be aware of the
    underlying parallelisation details.
    """

    # These attributes are exposed on the RemoteModel object; any other
    # accesses are forwarded to the Model objects running on the remote nodes.
    REMOTEMODEL_ATTRIBUTES = ('_client', '_view')

    def __init__(self, profile='mpi'):
        self._client = Client(profile=profile)
        self._view = self._client[:]
        self._view.block = True

        self._view.execute('from pyReef.Model.model import Model')
        self._view.execute('model = Model()')

        # Uncomment this to enable node debug logging to /tmp
        # self._view.apply(relog)

    def load_xml(self, filename, verbose=False):
        try:
            cwd = os.getcwd()
            self._view.execute('import os')
            self._view.execute('os.chdir("%s")' % cwd)
            self._view.execute('model.load_xml(filename="%s", verbose=%s)' % (filename, verbose))
        except Exception, e:
            import pdb; pdb.set_trace()

    def run_to_time(self, tEnd):
        try:
            self._view.execute('model.run_to_time(%s)' % tEnd)
        except Exception, e:
            import pdb; pdb.set_trace()

    def ncpus(self):
        """Return the number of CPUs used to generate the results."""
        return len(self._view)

    def __getattr__(self, name):
        """If we don't define an attribute locally, read its value from node 0"""
        return self._client[0]['model.%s' % name]

    def __setattr__(self, name, value):
        """If we don't define an attribute locally, write its value to all nodes"""

        if name in RemoteModel.REMOTEMODEL_ATTRIBUTES:
            self.__dict__[name] = value
        else:
            self._view['model.%s' % name] = value
