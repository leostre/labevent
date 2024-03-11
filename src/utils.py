def wrong_arguments_display(func):
    def wrap(self, *args, **kwargs):
        res = None
        try:
            res = getattr(self, func)(*args, **kwargs)
        except TypeError as err:
            err.args = (f'''Wrong arguments were passed to {func.__name__} of {str(type(self)).split('.')[-1]}
            Read the docs: 
            {self._doc()}''',)
            raise 
        finally:
            return res
    return wrap

