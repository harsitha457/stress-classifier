from django.db import models

class datainsert(models.Model):
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    city = models.CharField(max_length=100)
    class Meta:
        db_table = 'form'


# Create your models here.
